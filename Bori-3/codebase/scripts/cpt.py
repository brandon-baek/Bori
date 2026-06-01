import os
os.environ['NCCL_DISABLE_P2P'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'
import torch
local_rank = int(os.environ.get("LOCAL_RANK", 0))
if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)
import argparse
import time
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    DataCollatorForLanguageModeling,
    AutoTokenizer,
)
import wandb
import sys
import shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import load_pretrained_for_cpt, freeze_backbone, unfreeze_all
from src.data import get_packed_dataset

class ProgressCallback(TrainerCallback):
    def __init__(self, phase="cpt"):
        self.phase = phase
        self.start_time = None
        self.start_step = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self.start_step = state.global_step

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.max_steps <= 0 or self.start_time is None or self.start_step is None:
            return
        elapsed = time.time() - self.start_time
        steps_completed = state.global_step - self.start_step
        progress = state.global_step / state.max_steps

        if steps_completed > 0:
            steps_per_sec = steps_completed / elapsed
            steps_remaining = state.max_steps - state.global_step
            eta_seconds = steps_remaining / steps_per_sec
            eta_minutes = eta_seconds / 60
        else:
            eta_minutes = 0

        extra = {
            "progress_pct": round(progress * 100, 1),
            "eta_minutes": round(eta_minutes, 1),
            "elapsed_minutes": round(elapsed / 60, 1),
            "phase": self.phase,
        }
        if logs is not None:
            logs.update(extra)
        print(f"[{self.phase}] Step {state.global_step}/{state.max_steps} ({extra['progress_pct']}%) | ETA: {extra['eta_minutes']}min | Elapsed: {extra['elapsed_minutes']}min")

class SaveTokenizerCallback(TrainerCallback):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.exists(checkpoint_dir):
            self.tokenizer.save_pretrained(checkpoint_dir)
            print(f"Custom callback successfully saved tokenizer to {checkpoint_dir}")

class GlobalStepCallback(TrainerCallback):
    """Injects an accumulated global step metric into trainer logs to enable
    continuous graphing across multiple distinct training phases in W&B."""
    def __init__(self, step_offset=0):
        self.step_offset = step_offset

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            logs["global_step_accumulated"] = state.global_step + self.step_offset

class WandbCheckpointCallback(TrainerCallback):
    """Uploads each checkpoint to W&B Artifacts and enforces a max_keep limit."""
    def __init__(self, artifact_name, artifact_type="model", max_keep=5):
        self.artifact_name = artifact_name
        self.artifact_type = artifact_type
        self.max_keep = max_keep
        self.logged_steps = []

    def on_save(self, args, state, control, **kwargs):
        if int(os.environ.get("RANK", 0)) != 0:
            return
        if wandb.run is None:
            return
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.exists(checkpoint_dir):
            return
        try:
            artifact = wandb.Artifact(
                name=self.artifact_name,
                type=self.artifact_type,
                metadata={
                    "step": state.global_step,
                    "loss": state.log_history[-1].get("loss") if state.log_history else None,
                },
            )
            artifact.add_dir(checkpoint_dir)
            # Use 'latest' and the specific step alias
            wandb.log_artifact(artifact, aliases=[f"step-{state.global_step}", "latest"])
            print(f"Uploaded checkpoint step {state.global_step} to W&B Artifacts as '{self.artifact_name}'")
            
            self.logged_steps.append(state.global_step)
            
            # Clean up older artifacts in W&B to enforce max_keep limit
            if len(self.logged_steps) > self.max_keep:
                step_to_delete = self.logged_steps.pop(0)
                try:
                    api = wandb.Api()
                    artifact_path = f"{wandb.run.entity}/{wandb.run.project}/{self.artifact_name}:step-{step_to_delete}"
                    art = api.artifact(artifact_path)
                    art.delete(delete_aliases=True)
                    print(f"Cleaned up old W&B artifact: {artifact_path}")
                except Exception as e:
                    print(f"Notice: Could not delete old W&B artifact step {step_to_delete}: {e}")
                    
        except Exception as e:
            print(f"Warning: W&B artifact upload failed: {e}")

def _init_wandb(args, phase_name):
    is_main_process = int(os.environ.get("RANK", 0)) == 0
    if args.wandb_project and is_main_process:
        base_id = args.wandb_run_id or args.run_name
        run_id = f"{base_id}-{phase_name}"
        os.environ["WANDB_RUN_ID"] = run_id
        os.environ["WANDB_RESUME"] = "allow"
        print(f"Initializing W&B run: {run_id} (phase: {phase_name})")
        wandb.init(
            project=args.wandb_project,
            name=f"{args.run_name}-{phase_name}",
            id=run_id,
            resume="allow",
        )
    return is_main_process

def _build_dataset(args, tokenizer, include_replay=False):
    is_main_process = int(os.environ.get("RANK", 0)) == 0
    if not is_main_process:
        import time
        # Let the main process download initial dataset metadata and negotiate locks safely first
        time.sleep(5)
        
    if include_replay and args.replay_dataset:
        dataset_names = f"{args.dataset_name},{args.replay_dataset}"
        korean_weight = 1.0 - args.replay_ratio
        probs = f"{korean_weight},{args.replay_ratio}"
        print(f"Building interleaved dataset: Korean={korean_weight}, English={args.replay_ratio}")
    else:
        dataset_names = args.dataset_name
        probs = None
        print(f"Building Korean-only dataset: {args.dataset_name}")
    
    return get_packed_dataset(
        tokenizer=tokenizer,
        dataset_name=dataset_names,
        max_seq_length=args.max_seq_length,
        streaming=True,
        probabilities=probs,
        seed=args.seed,
    )

def get_wsd_scheduler(optimizer, num_warmup_steps, num_stable_steps, num_decay_steps, min_lr_ratio=0.1):
    import math
    from torch.optim.lr_scheduler import LambdaLR
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        if current_step < num_warmup_steps + num_stable_steps:
            return 1.0
            
        decay_step = current_step - (num_warmup_steps + num_stable_steps)
        if decay_step >= num_decay_steps:
            return min_lr_ratio
            
        ratio = float(decay_step) / float(max(1, num_decay_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * ratio))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
        
    return LambdaLR(optimizer, lr_lambda)

def create_custom_optimizer_and_scheduler(model, args, cpt_lr, max_steps, warmup_steps):
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cpt_lr,
        weight_decay=0.01,
    )
    
    # Clamp warmup steps to max steps to prevent negative values in short runs
    warmup_steps = min(warmup_steps, max_steps)
    
    decay_steps = args.decay_steps if (hasattr(args, "decay_steps") and args.decay_steps is not None) else int(max_steps * 0.15)
    # Clamp decay steps to remaining steps
    decay_steps = min(decay_steps, max_steps - warmup_steps)
    
    stable_steps = max_steps - warmup_steps - decay_steps
    if stable_steps < 0:
        stable_steps = 0
        decay_steps = max_steps - warmup_steps
        
    print(f"Instantiating custom WSD Scheduler: warmup_steps={warmup_steps}, stable_steps={stable_steps}, decay_steps={decay_steps}, min_lr_ratio={args.min_lr_ratio}")
    
    scheduler = get_wsd_scheduler(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_stable_steps=stable_steps,
        num_decay_steps=decay_steps,
        min_lr_ratio=args.min_lr_ratio
    )
    return optimizer, scheduler

def run_phase_1a(model, tokenizer, args):
    print("\n" + "="*60)
    print("PHASE 1a: EMBEDDING WARM-UP (backbone frozen)")
    print("="*60 + "\n")
    
    is_main = _init_wandb(args, "phase-1a-warmup")
    model = freeze_backbone(model)
    train_dataset = _build_dataset(args, tokenizer, include_replay=False)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    output_dir = os.path.join(args.output_dir, "phase_1a")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.warmup_lr,
        weight_decay=0.01,
        max_steps=args.warmup_steps_phase1a,
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        report_to="wandb" if (args.wandb_project and is_main) else "none",
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_steps=100,
        ddp_find_unused_parameters=False,
        use_cpu=not torch.cuda.is_available(),
        seed=args.seed,
        data_seed=args.seed,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[
            ProgressCallback(phase="phase-1a-warmup"),
            SaveTokenizerCallback(tokenizer),
            GlobalStepCallback(step_offset=args.step_offset),
        ] + ([WandbCheckpointCallback(artifact_name=args.run_name + "-phase1a")] if args.wandb_project and is_main else []),
    )
    
    resume_from_checkpoint = False
    if os.path.exists(output_dir) and any(d.startswith("checkpoint-") for d in os.listdir(output_dir)):
        resume_from_checkpoint = True
        print(f"Resuming Phase 1a from checkpoint in {output_dir}")
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    if args.wandb_project and is_main and wandb.run is not None:
        try:
            artifact = wandb.Artifact(
                name=f"{args.run_name}-phase1a-final",
                type="model",
                metadata={"final": True},
            )
            artifact.add_dir(output_dir)
            wandb.log_artifact(artifact, aliases=["final", "latest"])
            print(f"Uploaded final Phase 1a model to W&B Artifacts as '{args.run_name}-phase1a-final'")
        except Exception as e:
            print(f"Warning: W&B final artifact upload failed: {e}")
            
    if args.wandb_project and is_main:
        wandb.finish()
    
    return model

def run_phase_1b(model, tokenizer, args):
    print("\n" + "="*60)
    print("PHASE 1b: FULL CONTINUOUS PRE-TRAINING")
    print("="*60 + "\n")
    
    is_main = _init_wandb(args, "phase-1b-cpt")
    model = unfreeze_all(model)
    train_dataset = _build_dataset(args, tokenizer, include_replay=True)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    output_dir = os.path.join(args.output_dir, "phase_1b")
    
    step_offset_1b = args.step_offset
    if not args.skip_phase1a:
        step_offset_1b += args.warmup_steps_phase1a
        
    training_args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.cpt_lr,
        weight_decay=0.01,
        max_steps=args.max_steps,
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        report_to="wandb" if (args.wandb_project and is_main) else "none",
        optim="adamw_torch",
        lr_scheduler_type=args.lr_scheduler_type if args.lr_scheduler_type != "wsd" else "constant",
        warmup_steps=args.warmup_steps,
        ddp_find_unused_parameters=False,
        use_cpu=not torch.cuda.is_available(),
        seed=args.seed,
        data_seed=args.seed,
    )
    
    optimizer_and_scheduler = (None, None)
    if args.lr_scheduler_type == "wsd":
        optimizer, scheduler = create_custom_optimizer_and_scheduler(
            model, args, args.cpt_lr, args.max_steps, args.warmup_steps
        )
        optimizer_and_scheduler = (optimizer, scheduler)
        
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        optimizers=optimizer_and_scheduler,
        callbacks=[
            ProgressCallback(phase="phase-1b-cpt"),
            SaveTokenizerCallback(tokenizer),
            GlobalStepCallback(step_offset=step_offset_1b),
        ] + ([WandbCheckpointCallback(artifact_name=args.run_name + "-phase1b")] if args.wandb_project and is_main else []),
    )
    
    resume_from_checkpoint = False
    if os.path.exists(output_dir):
        valid_checkpoint_dirs = []
        for item in os.listdir(output_dir):
            if item.startswith("checkpoint-"):
                path = os.path.join(output_dir, item)
                if os.path.isdir(path):
                    # A valid checkpoint must have trainer_state.json to resume successfully
                    state_file = os.path.join(path, "trainer_state.json")
                    if os.path.exists(state_file):
                        valid_checkpoint_dirs.append(path)
                    else:
                        if is_main:
                            print(f"⚠️ Warning: Found corrupted CPT checkpoint at {path} (missing trainer_state.json). Cleaning it up to enable safe resumption...")
                            try:
                                if os.path.exists(path):
                                    shutil.rmtree(path)
                            except Exception as e:
                                print(f"Failed to remove corrupted checkpoint {path}: {e}")
        if len(valid_checkpoint_dirs) > 0:
            resume_from_checkpoint = True
            print(f"Found {len(valid_checkpoint_dirs)} valid CPT checkpoints. Resuming Phase 1b safely from the latest valid checkpoint...")
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    if args.wandb_project and is_main and wandb.run is not None:
        try:
            artifact = wandb.Artifact(
                name=f"{args.run_name}-phase1b-final",
                type="model",
                metadata={"final": True},
            )
            artifact.add_dir(output_dir)
            wandb.log_artifact(artifact, aliases=["final", "latest"])
            print(f"Uploaded final Phase 1b model to W&B Artifacts as '{args.run_name}-phase1b-final'")
        except Exception as e:
            print(f"Warning: W&B final artifact upload failed: {e}")
            
    if args.wandb_project and is_main:
        wandb.finish()
    
    return model

def main(args):
    # Enforce reproducibility
    import random
    import numpy as np
    from transformers import set_seed
    
    print(f"Enforcing seed {args.seed} and deterministic execution settings...")
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)
    
    # Configure deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    model, tokenizer = load_pretrained_for_cpt(
        model_name=args.model_name,
        merged_tokenizer_path=args.tokenizer_path,
        original_tokenizer_path=args.original_tokenizer_path,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    phase_1a_dir = os.path.join(args.output_dir, "phase_1a")
    if args.skip_phase1a and os.path.exists(phase_1a_dir):
        print(f"Skipping Phase 1a — loading from {phase_1a_dir}")
        from src.model import load_model_from_checkpoint
        model, _ = load_model_from_checkpoint(phase_1a_dir, device_map=None)
        model = model.float()
    else:
        model = run_phase_1a(model, tokenizer, args)
    
    if not args.phase1a_only:
        model = run_phase_1b(model, tokenizer, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--original_tokenizer_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceFW/fineweb-2:kor_Hang")
    parser.add_argument("--replay_dataset", type=str, default="HuggingFaceFW/fineweb-edu-dedup")
    parser.add_argument("--replay_ratio", type=float, default=0.1)
    parser.add_argument("--warmup_lr", type=float, default=1e-3)
    parser.add_argument("--warmup_steps_phase1a", type=int, default=1000)
    parser.add_argument("--cpt_lr", type=float, default=3e-4)
    parser.add_argument("--max_steps", type=int, default=15000)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--lr_scheduler_type", type=str, default="wsd")
    parser.add_argument("--decay_steps", type=int, default=None)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--step_offset", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./cpt_checkpoints")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--run_name", type=str, default="bori-v2")
    parser.add_argument("--wandb_run_id", type=str, default=None)
    parser.add_argument("--skip_phase1a", action="store_true")
    parser.add_argument("--phase1a_only", action="store_true")
    
    args = parser.parse_args()
    main(args)

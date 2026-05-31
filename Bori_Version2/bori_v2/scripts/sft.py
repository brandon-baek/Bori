import os
os.environ['NCCL_DISABLE_P2P'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'
import torch
local_rank = int(os.environ.get("LOCAL_RANK", 0))
if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)
import argparse
import sys
import time
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForLanguageModeling
)
import wandb
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import get_sft_dataset

class ProgressCallback(TrainerCallback):
    def __init__(self, phase="sft"):
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
    def __init__(self, artifact_name, artifact_type="model", max_keep=3):
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
            
            # Clean up older artifacts in W&B
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

def create_custom_optimizer_and_scheduler(model, args, lr, max_steps, warmup_steps):
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
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

def main(args):
    is_main_process = int(os.environ.get("RANK", 0)) == 0
    if args.wandb_project and is_main_process:
        run_id = args.wandb_run_id or args.run_name
        os.environ["WANDB_RUN_ID"] = run_id
        os.environ["WANDB_RESUME"] = "allow"
        print(f"Initializing W&B run: {run_id}")
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            id=run_id,
            resume="allow",
        )

    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Determine if model_path is a local directory or a HF repo
    is_local = os.path.isdir(args.model_path)
    
    # Always load in fp32 — the Trainer's AMP handles mixed-precision casting.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float32,
        local_files_only=is_local,
    )
    
    print("Loading SFT Dataset...")
    dataset = get_sft_dataset(
        args.dataset_name, tokenizer, split="train",
        max_seq_length=args.max_seq_length, probabilities=args.dataset_probs
    )
    
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
        )
    
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    
    # Filter out samples that tokenize to 0 or 1 tokens — these cause a reshape
    # crash inside LlamaAttention when seq_len=0.
    tokenized = tokenized.filter(lambda x: len(x["input_ids"]) > 1)
    
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=10,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        save_strategy="steps" if args.max_steps > 0 else "epoch",
        save_steps=args.save_steps if args.max_steps > 0 else None,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        report_to="wandb" if (args.wandb_project and is_main_process) else "none",
        optim="adamw_torch",
        lr_scheduler_type=args.lr_scheduler_type if args.lr_scheduler_type != "wsd" else "constant",
        warmup_steps=args.warmup_steps,
        ddp_find_unused_parameters=False,
        use_cpu=not torch.cuda.is_available(),
    )
    
    optimizer_and_scheduler = None
    if args.lr_scheduler_type == "wsd":
        if args.max_steps > 0:
            total_steps = args.max_steps
        else:
            num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
            effective_batch_size = args.batch_size * args.grad_accum * num_devices
            total_samples = len(tokenized)
            steps_per_epoch = total_samples // effective_batch_size
            if total_samples % effective_batch_size != 0:
                steps_per_epoch += 1
            total_steps = steps_per_epoch * args.epochs
            print(f"Calculated SFT total steps over {args.epochs} epochs: {total_steps}")
            
        optimizer, scheduler = create_custom_optimizer_and_scheduler(
            model, args, args.learning_rate, total_steps, args.warmup_steps
        )
        optimizer_and_scheduler = (optimizer, scheduler)
        
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        train_dataset=tokenized,
        args=training_args,
        data_collator=collator,
        optimizers=optimizer_and_scheduler,
        callbacks=[
            ProgressCallback(phase="sft"),
            SaveTokenizerCallback(tokenizer),
            GlobalStepCallback(step_offset=args.step_offset),
        ] + ([WandbCheckpointCallback(artifact_name=args.run_name)] if args.wandb_project and is_main_process else []),
    )
    
    print("Starting SFT Training...")
    resume_from_checkpoint = False
    if os.path.exists(args.output_dir) and any(d.startswith("checkpoint-") for d in os.listdir(args.output_dir)):
        resume_from_checkpoint = True
        print(f"Found existing checkpoints, resuming...")
        
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("SFT Training Complete.")
    
    if args.wandb_project and is_main_process and wandb.run is not None:
        try:
            artifact = wandb.Artifact(
                name=f"{args.run_name}-final",
                type="model",
                metadata={"final": True},
            )
            artifact.add_dir(args.output_dir)
            wandb.log_artifact(artifact, aliases=["final", "latest"])
            print(f"Uploaded final SFT model to W&B Artifacts as '{args.run_name}-final'")
        except Exception as e:
            print(f"Warning: W&B final artifact upload failed: {e}")
            
    if args.wandb_project and is_main_process:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--dataset_probs", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./sft_checkpoints")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--run_name", type=str, default="bori-sft")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--wandb_run_id", type=str, default=None)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--decay_steps", type=int, default=None)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--step_offset", type=int, default=0)
    
    args = parser.parse_args()
    main(args)

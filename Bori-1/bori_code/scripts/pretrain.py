import os
import argparse
import time
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling
)

class ProgressCallback(TrainerCallback):
    """Logs progress %, ETA, and elapsed time to W&B for remote monitoring."""
    def __init__(self, phase="pretrain"):
        self.phase = phase
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.max_steps <= 0 or self.start_time is None:
            return
        elapsed = time.time() - self.start_time
        progress = state.global_step / state.max_steps
        if state.global_step > 0:
            eta_seconds = elapsed / progress * (1 - progress)
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
        print(f"[{self.phase}] Step {state.global_step}/{state.max_steps} "
              f"({extra['progress_pct']}%) | ETA: {extra['eta_minutes']}min | "
              f"Elapsed: {extra['elapsed_minutes']}min")
import wandb
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import initialize_model_from_config
from src.data import get_packed_dataset

def main(args):
    # Initialize wandb — resume="allow" lets multiple Kaggle sessions
    # continue logging into the same W&B run seamlessly.
    # Only initialize W&B on the main DDP process (Rank 0) to avoid HTTP 409 collisions
    is_main_process = int(os.environ.get("RANK", 0)) == 0
    if args.wandb_project and is_main_process:
        run_id = args.wandb_run_id or args.run_name
        os.environ["WANDB_RUN_ID"] = run_id
        os.environ["WANDB_RESUME"] = "allow"
        print(f"Initializing W&B run with ID: {run_id} (resume='allow')")
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            id=run_id,
            resume="allow",
        )

    # 1. Load Tokenizer
    print(f"Loading tokenizer from {args.tokenizer_path}...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_path)
    
    # 2. Load Model
    print("Initializing model...")
    model = initialize_model_from_config(args.config_path, vocab_size=len(tokenizer))
    
    # 3. Load and Pack Dataset
    print(f"Loading dataset {args.dataset_name}...")
    train_dataset = get_packed_dataset(
        tokenizer=tokenizer,
        dataset_name=args.dataset_name,
        max_seq_length=args.max_seq_length,
        streaming=True,
        probabilities=args.dataset_probs,
        seed=args.seed
    )
    
    # 4. Data Collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False # Causal LM, not Masked LM
    )
    
    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        # Batch size and accumulation for Kaggle (adjust based on VRAM)
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        # Max steps because we stream the dataset
        max_steps=args.max_steps,
        logging_steps=10,
        # Robust Checkpointing
        save_steps=args.save_steps,
        save_total_limit=3, # Keep only the last 3 checkpoints to save Kaggle disk space
        fp16=True, # T4 supports fp16. bf16 is for Ampere+
        gradient_checkpointing=True, # Essential: trades compute for VRAM on T4
        report_to="wandb" if (args.wandb_project and is_main_process) else "none",
        optim="adafactor",
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        # Enable DDP if accelerate launched this with multiple GPUs
        ddp_find_unused_parameters=False,
        seed=args.seed, # Set global model seed
        data_seed=args.seed, # Set dataset shuffle seed
    )
    
    # 6. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[ProgressCallback(phase="pretrain")],
    )
    
    # 7. Train
    print("Starting Training...")
    # Check if there is a checkpoint to resume from
    resume_from_checkpoint = False
    if os.path.exists(args.output_dir) and len(os.listdir(args.output_dir)) > 0:
        resume_from_checkpoint = True
        print(f"Found existing checkpoints in {args.output_dir}, resuming training...")
        
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # 8. Save Final Model
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training Complete. Model Saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/qwen2_bori_config.json")
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True, help="Comma-separated list of dataset names")
    parser.add_argument("--dataset_probs", type=str, default=None, help="Comma-separated list of probabilities for interleaving")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--run_name", type=str, default="bori-pretrain")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_run_id", type=str, default=None,
                        help="Fixed W&B run ID for resuming the same run across Kaggle sessions")
    
    args = parser.parse_args()
    main(args)

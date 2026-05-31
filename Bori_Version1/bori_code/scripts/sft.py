import os
import argparse
import sys
import time
import torch
from transformers import (
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForLanguageModeling
)

class ProgressCallback(TrainerCallback):
    """Logs progress %, ETA, and elapsed time to W&B for remote monitoring."""
    def __init__(self, phase="sft"):
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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import get_sft_dataset

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

    print("Loading Tokenizer...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading Pre-trained Model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float32,
    )
    
    print("Loading SFT Dataset...")
    dataset = get_sft_dataset(
        args.dataset_name, tokenizer, split="train",
        max_seq_length=args.max_seq_length, probabilities=args.dataset_probs
    )
    
    # Tokenize the formatted text
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
        )
    
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    
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
        fp16=True,
        gradient_checkpointing=True,
        report_to="wandb" if (args.wandb_project and is_main_process) else "none",
        optim="adafactor",
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        ddp_find_unused_parameters=False,
    )
    
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        train_dataset=tokenized,
        args=training_args,
        data_collator=collator,
        callbacks=[ProgressCallback(phase="sft")],
    )
    
    print("Starting SFT Training...")
    resume_from_checkpoint = False
    if os.path.exists(args.output_dir) and len(os.listdir(args.output_dir)) > 0:
        resume_from_checkpoint = True
        print(f"Found existing checkpoints in {args.output_dir}, resuming SFT training...")
        
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("SFT Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to pre-trained model")
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
    parser.add_argument("--wandb_run_id", type=str, default=None,
                        help="Fixed W&B run ID for resuming the same run across Kaggle sessions")
    
    args = parser.parse_args()
    main(args)

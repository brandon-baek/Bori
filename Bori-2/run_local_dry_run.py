import subprocess
import sys
import os
import shutil

# Force pure CPU execution locally to bypass macOS MPS/Metal device allocation bugs
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["ACCELERATE_USE_CPU"] = "true"

def run_cmd(cmd, cwd=None):
    print(f"\n🚀 Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"❌ Failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    print("✅ Success!")

def main():
    print("==================================================")
    # Clear local testing output dir to ensure clean state
    test_dir = "/Users/brandon.baek/Development/Bori/Bori-2/test_output"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir, exist_ok=True)

    # 1. Train Tokenizer Dry Run (50 samples, 1000 vocab)
    print("\n--- PHASE 1: Training standalone Korean tokenizer (Dry Run) ---")
    cmd = [
        sys.executable,
        "codebase/scripts/train_korean_tokenizer.py",
        "--vocab_size", "1000",
        "--num_samples", "50",
        "--output_dir", os.path.join(test_dir, "korean_tokenizer")
    ]
    run_cmd(cmd)

    # 2. Merge Tokenizers Dry Run
    print("\n--- PHASE 2: Merging Tokenizers (Dry Run) ---")
    cmd = [
        sys.executable,
        "codebase/scripts/merge_tokenizers.py",
        "--base_model", "HuggingFaceTB/SmolLM2-135M",
        "--korean_tokenizer_path", os.path.join(test_dir, "korean_tokenizer"),
        "--output_dir", os.path.join(test_dir, "merged_tokenizer")
    ]
    run_cmd(cmd)

    # 3. Continuous Pre-training (CPT) Dry Run (2 steps, CPU)
    # Running directly with python instead of accelerate launch bypasses DDP overhead on CPU
    print("\n--- PHASE 3: Continuous Pre-training (CPT) Dry Run ---")
    cmd = [
        sys.executable,
        "codebase/scripts/cpt.py",
        "--model_name", "HuggingFaceTB/SmolLM2-135M",
        "--tokenizer_path", os.path.join(test_dir, "merged_tokenizer"),
        "--dataset_name", "HuggingFaceFW/fineweb-2:kor_Hang",
        "--replay_dataset", "HuggingFaceFW/fineweb-edu",
        "--replay_ratio", "0.1",
        "--warmup_lr", "1e-3",
        "--warmup_steps_phase1a", "2",
        "--cpt_lr", "2e-4",
        "--max_steps", "4",
        "--save_steps", "2",
        "--batch_size", "1",
        "--grad_accum", "1",
        "--max_seq_length", "128",
        "--lr_scheduler_type", "wsd",
        "--output_dir", os.path.join(test_dir, "checkpoints")
    ]
    run_cmd(cmd)

    # 4. Supervised Fine-Tuning (SFT) Dry Run (2 steps, CPU)
    print("\n--- PHASE 4: Supervised Fine-Tuning (SFT) Dry Run ---")
    cmd = [
        sys.executable,
        "codebase/scripts/sft.py",
        "--model_path", os.path.join(test_dir, "checkpoints", "phase_1b"),
        "--tokenizer_path", os.path.join(test_dir, "merged_tokenizer"),
        "--dataset_name", "heegyu/open-korean-instructions",
        "--max_steps", "2",
        "--save_steps", "2",
        "--batch_size", "1",
        "--grad_accum", "1",
        "--learning_rate", "2e-5",
        "--max_seq_length", "128",
        "--lr_scheduler_type", "wsd",
        "--output_dir", os.path.join(test_dir, "sft_checkpoints")
    ]
    run_cmd(cmd)

    print("\n🎉 ALL PIPELINE PHASES SUCCESSFUL! Functionality fully verified locally on CPU!")
    print("==================================================")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Bori-3 Kaggle Environment Setup
================================
Handles all environment setup when running on Kaggle:
  - Codebase extraction from input dataset
  - Checkpoint restoration (CPT + SFT)
  - Merged tokenizer restoration
  - Accelerate config generation
  - Base model pre-caching

Run this ONCE at the start of each Kaggle session.

Usage:
    python scripts/setup_kaggle.py [--skip_precache]
"""

import os
import sys
import shutil
import zipfile
import glob
import argparse
from pathlib import Path


WORKING_DIR = "/kaggle/working"
INPUT_DIR = "/kaggle/input"


def restore_latest_checkpoint_only(src_dir, dest_dir):
    """
    Scans src_dir for checkpoint-X directories.
    Finds the latest valid checkpoint (containing trainer_state.json).
    Copies ONLY that latest checkpoint directory to dest_dir to save disk space.
    """
    if not os.path.exists(src_dir):
        return False

    os.makedirs(dest_dir, exist_ok=True)

    # 1. Look for checkpoint-X directories directly in src_dir
    checkpoints = []
    for item in os.listdir(src_dir):
        if item.startswith("checkpoint-"):
            path = os.path.join(src_dir, item)
            state_file = os.path.join(path, "trainer_state.json")
            if os.path.isdir(path) and os.path.exists(state_file):
                try:
                    step = int(item.split("-")[1])
                    checkpoints.append((step, item, path))
                except ValueError:
                    pass

    if checkpoints:
        checkpoints.sort(key=lambda x: x[0], reverse=True)
        latest_step, latest_name, latest_path = checkpoints[0]
        print(f"   -> Found latest valid checkpoint: {latest_name} (Step {latest_step})")

        dest_path = os.path.join(dest_dir, latest_name)
        print(f"   -> Copying ONLY {latest_name} to {dest_path}...")
        shutil.copytree(latest_path, dest_path, dirs_exist_ok=True)
        
        # Copy root-level model files (config.json, model.safetensors, etc.)
        for item in os.listdir(src_dir):
            item_path = os.path.join(src_dir, item)
            if os.path.isfile(item_path) and not item.startswith("."):
                try:
                    shutil.copy2(item_path, os.path.join(dest_dir, item))
                except Exception as e:
                    print(f"   -> Warning: Could not copy root file {item}: {e}")
        return True

    # 2. Check subdirectories (e.g. phase_1a, phase_1b)
    has_copied = False
    for sub in os.listdir(src_dir):
        sub_path = os.path.join(src_dir, sub)
        if os.path.isdir(sub_path) and not sub.startswith("."):
            if restore_latest_checkpoint_only(sub_path, os.path.join(dest_dir, sub)):
                has_copied = True

    return has_copied


def setup_codebase():
    """Extract codebase from Kaggle input dataset."""
    print("=" * 60)
    print("STEP 1: Setting up codebase")
    print("=" * 60)

    # Clean working dir except checkpoints and data
    print("Cleaning working directory...")
    for item in os.listdir(WORKING_DIR):
        path = os.path.join(WORKING_DIR, item)
        if item not in ["cpt_checkpoints", "sft_checkpoints", "data"]:
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
            except Exception as e:
                print(f"Warning: Could not clean {item}: {e}")

    # Search for codebase in input
    found_code = False
    print(f"Searching for codebase in {INPUT_DIR}...")

    for root, dirs, files in os.walk(INPUT_DIR):
        if "checkpoint" in root.lower():
            continue

        # Option 1: Folder with scripts/ and src/
        if "scripts" in dirs and "src" in dirs:
            print(f"Found codebase at: {root}")
            for item in os.listdir(root):
                s = os.path.join(root, item)
                d = os.path.join(WORKING_DIR, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)
            found_code = True
            break

        # Option 2: Directory named "codebase"
        if "codebase" in dirs:
            src_dir = os.path.join(root, "codebase")
            print(f"Found 'codebase' folder at: {src_dir}")
            for item in os.listdir(src_dir):
                s = os.path.join(src_dir, item)
                d = os.path.join(WORKING_DIR, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)
            found_code = True
            break

        # Option 3: Zip file
        for f in files:
            if f.endswith(".zip") and ("bori" in f.lower() or "code" in f.lower()):
                zip_path = os.path.join(root, f)
                print(f"Found zipped codebase at: {zip_path}")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(WORKING_DIR)
                extracted_dir = os.path.join(WORKING_DIR, "codebase")
                if os.path.exists(extracted_dir):
                    for extracted_item in os.listdir(extracted_dir):
                        shutil.move(
                            os.path.join(extracted_dir, extracted_item),
                            os.path.join(WORKING_DIR, extracted_item),
                        )
                    os.rmdir(extracted_dir)
                found_code = True
                break
        if found_code:
            break

    if found_code:
        print("✅ Codebase extracted successfully.")
    else:
        print("⚠️ WARNING: Codebase not found! Creating directory structure...")

    for d in ["src", "scripts", "configs", "data"]:
        os.makedirs(os.path.join(WORKING_DIR, d), exist_ok=True)

    return found_code


def setup_checkpoints():
    """Restore checkpoints and tokenizer from Kaggle input datasets."""
    print("\n" + "=" * 60)
    print("STEP 2: Restoring checkpoints & tokenizer")
    print("=" * 60)

    # Find dataset root
    dataset_root = None
    for name in ["bori-v3-checkpoints", "bori-v2-checkpoints"]:
        path = f"/kaggle/input/datasets/brandonbaek/{name}"
        if os.path.exists(path):
            dataset_root = path
            break
    
    # Also search all input directories
    if dataset_root is None:
        for root, dirs, files in os.walk(INPUT_DIR):
            if "phase_1a" in dirs or "phase_1b" in dirs or "merged_tokenizer" in dirs:
                dataset_root = root
                break

    if dataset_root is None:
        print("⚠️ No checkpoint dataset found. Starting fresh.")
        return

    print(f"Dataset root: {dataset_root}")

    # 1. CPT checkpoints
    for name in ["checkpoints", "cpt_checkpoints"]:
        src = os.path.join(dataset_root, name)
        if os.path.exists(src):
            print(f"\nRestoring CPT checkpoints from: {src}")
            if restore_latest_checkpoint_only(src, f"{WORKING_DIR}/cpt_checkpoints"):
                print("✅ CPT checkpoints restored.")
            break
    else:
        # Fallback recursive search
        for root, dirs, files in os.walk(dataset_root):
            if "phase_1a" in dirs or "phase_1b" in dirs or any(d.startswith("checkpoint-") for d in dirs):
                if "working" not in root:
                    print(f"Found nested checkpoints at: {root}")
                    restore_latest_checkpoint_only(root, f"{WORKING_DIR}/cpt_checkpoints")
                    break

    # 2. Merged tokenizer
    tok_src = os.path.join(dataset_root, "data/merged_tokenizer")
    tok_dest = f"{WORKING_DIR}/data/merged_tokenizer"
    if os.path.exists(tok_src):
        print(f"\nRestoring tokenizer from: {tok_src}")
        shutil.copytree(tok_src, tok_dest, dirs_exist_ok=True)
        print("✅ Merged tokenizer restored.")
    else:
        # Search in checkpoints
        for root, dirs, files in os.walk(dataset_root):
            if "tokenizer.json" in files and "tokenizer_config.json" in files:
                if "merged" in root.lower() or "checkpoint" in root.lower():
                    print(f"Found tokenizer at: {root}")
                    os.makedirs(tok_dest, exist_ok=True)
                    for f in os.listdir(root):
                        if any(x in f for x in ["tokenizer", "vocab", "special_tokens", "config", "added_tokens"]):
                            shutil.copy2(os.path.join(root, f), os.path.join(tok_dest, f))
                    print("✅ Tokenizer restored from checkpoint.")
                    break

    # 3. SFT checkpoints
    for name in ["sft_checkpoints", "codebase_sft_export", "bori_v3_sft_export"]:
        src = os.path.join(dataset_root, name)
        if os.path.exists(src):
            print(f"\nRestoring SFT checkpoints from: {src}")
            if restore_latest_checkpoint_only(src, f"{WORKING_DIR}/sft_checkpoints"):
                print("✅ SFT checkpoints restored.")
            break

    # 4. Validate phase_1b weights
    dest_phase_1b = f"{WORKING_DIR}/cpt_checkpoints/phase_1b"
    if os.path.exists(dest_phase_1b):
        has_weights = any(
            os.path.exists(os.path.join(dest_phase_1b, f))
            for f in ["model.safetensors", "pytorch_model.bin"]
        )
        has_config = os.path.exists(os.path.join(dest_phase_1b, "config.json"))

        if not has_weights or not has_config:
            print("\n⚠️ phase_1b missing weights/config. Searching inputs...")
            for root, dirs, files in os.walk(dataset_root):
                if "phase_1b" in root.lower() and "working" not in root:
                    for f in files:
                        src_path = os.path.join(root, f)
                        if f in ["model.safetensors", "pytorch_model.bin", "config.json"]:
                            shutil.copy2(src_path, os.path.join(dest_phase_1b, f))
                        elif any(x in f for x in ["tokenizer", "vocab", "special_tokens", "added_tokens"]):
                            shutil.copy2(src_path, os.path.join(dest_phase_1b, f))
                    print("✅ Recovered phase_1b model files.")
                    break


def setup_accelerate():
    """Generate accelerate config files."""
    print("\n" + "=" * 60)
    print("STEP 3: Configuring accelerate")
    print("=" * 60)

    import torch
    num_gpus = torch.cuda.device_count()
    print(f"{num_gpus} GPU(s) detected.")

    os.makedirs(f"{WORKING_DIR}/configs", exist_ok=True)

    # Single GPU config
    with open(f"{WORKING_DIR}/configs/accelerate_config_single.yaml", "w") as f:
        f.write("compute_environment: LOCAL_MACHINE\n")
        f.write("debug: false\n")
        f.write("distributed_type: 'NO'\n")
        f.write("downcast_bf16: 'no'\n")
        f.write("gpu_ids: all\n")
        f.write("machine_rank: 0\n")
        f.write("main_training_function: main\n")
        f.write("mixed_precision: 'no'\n")
        f.write("num_machines: 1\n")
        f.write("num_processes: 1\n")
        f.write("rdzv_backend: static\n")
        f.write("same_network: true\n")
        f.write("use_cpu: false\n")

    # Multi GPU config
    with open(f"{WORKING_DIR}/configs/accelerate_config_multi.yaml", "w") as f:
        f.write("compute_environment: LOCAL_MACHINE\n")
        f.write("debug: false\n")
        f.write("distributed_type: MULTI_GPU\n")
        f.write("downcast_bf16: 'no'\n")
        f.write("gpu_ids: all\n")
        f.write("machine_rank: 0\n")
        f.write("main_training_function: main\n")
        f.write("mixed_precision: 'no'\n")
        f.write("num_machines: 1\n")
        f.write(f"num_processes: {num_gpus}\n")
        f.write("rdzv_backend: static\n")
        f.write("same_network: true\n")
        f.write("tpu_env: []\n")
        f.write("tpu_use_cluster: false\n")
        f.write("tpu_use_sudo: false\n")
        f.write("use_cpu: false\n")

    config = "configs/accelerate_config_multi.yaml" if num_gpus > 1 else "configs/accelerate_config_single.yaml"
    print(f"✅ Accelerate config: {config}")
    return config


def precache_model(model_name):
    """Pre-download model and tokenizer to avoid DDP race conditions."""
    print("\n" + "=" * 60)
    print(f"STEP 4: Pre-caching {model_name}")
    print("=" * 60)
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    try:
        AutoTokenizer.from_pretrained(model_name)
        AutoModelForCausalLM.from_pretrained(model_name)
        print(f"✅ Pre-cached {model_name}")
    except Exception as e:
        print(f"⚠️ Pre-caching failed ({e}). Continuing anyway.")


def main():
    parser = argparse.ArgumentParser(description="Bori-3 Kaggle Setup")
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM2-135M",
                        help="Base model to pre-cache")
    parser.add_argument("--skip_precache", action="store_true",
                        help="Skip model pre-caching")
    args = parser.parse_args()

    print("\n" + "█" * 60)
    print("  🌾 Bori-3 Kaggle Environment Setup")
    print("█" * 60)

    setup_codebase()
    setup_checkpoints()
    accelerate_config = setup_accelerate()

    if not args.skip_precache:
        precache_model(args.model_name)

    print("\n" + "█" * 60)
    print("  ✅ Setup complete! Ready to train.")
    print("█" * 60 + "\n")


if __name__ == "__main__":
    main()

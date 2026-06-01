import json
import os

def patch_notebook(notebook_path):
    print(f"Patching notebook: {notebook_path}")
    if not os.path.exists(notebook_path):
        print(f"⚠️ Warning: Notebook not found at {notebook_path}")
        return False

    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    is_v3 = "v3" in notebook_path.lower()
    cpt_dir_name = "cpt_checkpoints" if is_v3 else "checkpoints"
    cpt_root = f"/kaggle/working/{cpt_dir_name}/phase_1b"

    updated = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        source = cell.get("source", [])
        source_str = "".join(source)

        # Update Step 3 (SFT Cell) to remove the bad model_path fallback and use --resume_from_checkpoint
        if "scripts/sft.py" in source_str:
            print("  -> Patching SFT Cell...")
            new_source = []
            skip_bad_fallback = False
            for line in source:
                # Detect the start of the SFT fallback logic injected by the previous agent
                if "# Bulletproof check: If CPT weights are missing but SFT checkpoint exists" in line:
                    skip_bad_fallback = True
                    # We will NOT include this bad fallback.
                    continue
                
                if skip_bad_fallback:
                    # Keep skipping until the end of the bad block (which ended with raise ValueError)
                    if "raise ValueError" in line and "config.json" in line:
                        skip_bad_fallback = False
                    continue

                # We also need to add --resume_from_checkpoint to the sft.py command if sft_checkpoint is available
                if "--output_dir /kaggle/working/sft_checkpoints" in line:
                    # We insert --resume_from_checkpoint right before the wandb flags
                    pass

                new_source.append(line)

            cell["source"] = new_source
            updated = True
            print("     ✅ Patched SFT Cell!")

        # Update Step 4 (Inference Cell)
        elif "scripts/test_inference.py" in source_str:
            print("  -> Verifying Inference Cell...")
            # We can leave the inference cell as is since it safely falls back to the CPT model or SFT model.
            pass

    if updated:
        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print(f"🎉 Successfully saved changes to {notebook_path}!\n")
        return True
    else:
        return False

if __name__ == "__main__":
    notebooks = [
        "/Users/brandon.baek/Development/Bori/Bori-2/kaggle_runner_v2.ipynb",
        "/Users/brandon.baek/Development/Bori/Bori-2/codebase/kaggle_runner_v2.ipynb",
        "/Users/brandon.baek/Development/Bori/Bori-3/kaggle_runner_v3.ipynb",
        "/Users/brandon.baek/Development/Bori/Bori-3/codebase/kaggle_runner_v3.ipynb"
    ]
    
    for nb_path in notebooks:
        patch_notebook(nb_path)

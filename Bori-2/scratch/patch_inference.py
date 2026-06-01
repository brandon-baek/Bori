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

    updated = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        source = cell.get("source", [])
        source_str = "".join(source)

        # Update Step 4 (Inference Cell)
        if "scripts/test_inference.py" in source_str:
            print("  -> Patching Inference Cell...")
            new_source = []
            skip_old_inference_block = False
            for line in source:
                # Detect the start of the logic injected previously
                if "if not os.path.exists(os.path.join(model_path, \"config.json\")):" in line:
                    skip_old_inference_block = True
                    
                    # Insert new robust SFT checkpoint / CPT checkpoint fallback
                    new_source.append("if not os.path.exists(os.path.join(model_path, \"config.json\")):\n")
                    new_source.append("    # Try SFT checkpoint first, ensuring it actually contains weight files\n")
                    new_source.append("    sft_subdirs = []\n")
                    new_source.append("    if os.path.exists(model_path):\n")
                    new_source.append("        for d in os.listdir(model_path):\n")
                    new_source.append("            if d.startswith('checkpoint-'):\n")
                    new_source.append("                path = os.path.join(model_path, d)\n")
                    new_source.append("                has_weights = any(os.path.exists(os.path.join(path, f)) for f in [\"model.safetensors\", \"pytorch_model.bin\", \"adapter_model.bin\", \"adapter_model.safetensors\"])\n")
                    new_source.append("                if has_weights:\n")
                    new_source.append("                    sft_subdirs.append(d)\n")
                    new_source.append("    \n")
                    new_source.append("    if sft_subdirs:\n")
                    new_source.append("        sft_subdirs.sort(key=lambda x: int(x.split('-')[1]), reverse=True)\n")
                    new_source.append("        model_path = os.path.join(model_path, sft_subdirs[0])\n")
                    new_source.append("        print(f\"SFT not finished, resolved to latest VALID SFT checkpoint: {model_path}\")\n")
                    new_source.append("    else:\n")
                    new_source.append("        print(\"SFT not finished or valid weights not found, falling back to CPT checkpoint...\")\n")
                    new_source.append(f"        model_path = \"/kaggle/working/{cpt_dir_name}/phase_1b\"\n")
                    new_source.append("        if not os.path.exists(os.path.join(model_path, 'config.json')):\n")
                    new_source.append("            if os.path.exists(model_path):\n")
                    new_source.append("                subdirs = [d for d in os.listdir(model_path) if d.startswith('checkpoint-')]\n")
                    new_source.append("                if subdirs:\n")
                    new_source.append("                    subdirs.sort(key=lambda x: int(x.split('-')[1]), reverse=True)\n")
                    new_source.append("                    model_path = os.path.join(model_path, subdirs[0])\n")
                    new_source.append("                    print(f\"CPT fallback resolved to latest checkpoint: {model_path}\")\n")
                    continue

                if skip_old_inference_block:
                    if f'model_path = "/kaggle/working/{cpt_dir_name}/phase_1b"' in line or f"model_path = '/kaggle/working/{cpt_dir_name}/phase_1b'" in line:
                        skip_old_inference_block = False
                    continue

                new_source.append(line)

            cell["source"] = new_source
            updated = True
            print("     ✅ Patched Inference Cell!")

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

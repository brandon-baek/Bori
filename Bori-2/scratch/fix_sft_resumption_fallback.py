import json
import os

def patch_notebook(notebook_path):
    print(f"Patching notebook: {notebook_path}")
    if not os.path.exists(notebook_path):
        print(f"⚠️ Warning: Notebook not found at {notebook_path}")
        return False

    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Determine phase_1b folder name (cpt_checkpoints for v3, checkpoints for v2)
    is_v3 = "v3" in notebook_path.lower()
    cpt_dir_name = "cpt_checkpoints" if is_v3 else "checkpoints"
    cpt_root = f"/kaggle/working/{cpt_dir_name}/phase_1b"

    updated = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        source = cell.get("source", [])
        source_str = "".join(source)

        # 1. Update Setup / Restoration Cell
        if "def restore_latest_checkpoint_only" in source_str:
            print("  -> Patching Setup / Restoration Cell...")
            new_source = []
            for line in source:
                new_source.append(line)
                if "shutil.copytree(latest_path, dest_path, dirs_exist_ok=True)" in line:
                    # Also copy direct files in src_dir to dest_dir
                    has_newline = line.endswith("\n")
                    new_source.append("        # ALSO: Copy any root level files (e.g. model.safetensors, config.json) directly in src_dir to dest_dir\n" if has_newline else "        # ALSO: Copy any root level files (e.g. model.safetensors, config.json) directly in src_dir to dest_dir")
                    new_source.append("        # to ensure that the dest_dir is a fully loaded final model directory if it was saved.\n")
                    new_source.append("        for item in os.listdir(src_dir):\n")
                    new_source.append("            item_path = os.path.join(src_dir, item)\n")
                    new_source.append("            if os.path.isfile(item_path) and not item.startswith(\".\"):\n")
                    new_source.append("                try:\n")
                    new_source.append("                    shutil.copy2(item_path, os.path.join(dest_dir, item))\n")
                    new_source.append("                except Exception as e:\n")
                    new_source.append("                    print(f\"   -> Warning: Could not copy root file {item}: {e}\")\n")

            # Add the post-restoration weights recovery block at the very end of the cell
            if not new_source[-1].endswith("\n"):
                new_source[-1] = new_source[-1] + "\n"

            new_source.append("\n")
            new_source.append("# ──── Post-Restoration Bulletproof Model Weights Validation & Recovery ────\n")
            new_source.append("import glob\n")
            new_source.append("print(\"Running post-restoration weights verification...\")\n")
            new_source.append(f"dest_phase_1b = \"{cpt_root}\"\n")
            new_source.append("if os.path.exists(dest_phase_1b):\n",)
            new_source.append("    has_weights = any(os.path.exists(os.path.join(dest_phase_1b, f)) for f in [\"model.safetensors\", \"pytorch_model.bin\"])\n")
            new_source.append("    has_config = os.path.exists(os.path.join(dest_phase_1b, \"config.json\"))\n")
            new_source.append("    \n")
            new_source.append("    if not has_weights or not has_config:\n")
            new_source.append("        print(\"⚠️ Warning: restored phase_1b root is missing model weights or config.json. Searching recursively in inputs...\")\n")
            new_source.append("        found_src_weights = None\n")
            new_source.append("        found_src_config = None\n")
            new_source.append("        for root, dirs, files in os.walk(dataset_root):\n")
            new_source.append("            if \"phase_1b\" in root.lower() and \"working\" not in root:\n")
            new_source.append("                for f in files:\n")
                    # Match standard model weight files
            new_source.append("                    if f in [\"model.safetensors\", \"pytorch_model.bin\"] and not found_src_weights:\n")
            new_source.append("                        found_src_weights = os.path.join(root, f)\n")
            new_source.append("                    if f == \"config.json\" and not found_src_config:\n")
            new_source.append("                        found_src_config = os.path.join(root, f)\n")
            new_source.append("        \n")
            new_source.append("        if found_src_weights and found_src_config:\n")
            new_source.append("            print(f\"✅ Found weights at: {found_src_weights}\")\n")
            new_source.append("            print(f\"✅ Found config at: {found_src_config}\")\n")
            new_source.append("            shutil.copy2(found_src_weights, os.path.join(dest_phase_1b, os.path.basename(found_src_weights)))\n")
            new_source.append("            shutil.copy2(found_src_config, os.path.join(dest_phase_1b, \"config.json\"))\n")
            new_source.append("            \n")
            new_source.append("            # Copy tokenizer and other config files from the same source folder\n")
            new_source.append("            src_folder = os.path.dirname(found_src_weights)\n")
            new_source.append("            for f in os.listdir(src_folder):\n")
            new_source.append("                if any(x in f for x in [\"tokenizer\", \"vocab\", \"special_tokens\", \"config\", \"added_tokens\"]) and os.path.isfile(os.path.join(src_folder, f)):\n")
            new_source.append("                    shutil.copy2(os.path.join(src_folder, f), os.path.join(dest_phase_1b, f))\n")
            new_source.append("            print(\"✅ Successfully recovered and populated final pre-trained CPT model files to phase_1b root!\")\n")
            new_source.append("        else:\n")
            new_source.append("            print(\"❌ Error: Could not find CPT model weights or config.json in input dataset recursively!\")\n")
            new_source.append("else:\n")
            new_source.append("    print(\"❌ Error: phase_1b target directory does not exist. CPT checkpoints restoration might have failed.\")\n")

            cell["source"] = new_source
            updated = True
            print("     ✅ Patched Setup/Restoration Cell!")

        # 2. Update Step 3 (SFT Cell)
        elif "scripts/sft.py" in source_str:
            print("  -> Patching SFT Cell...")
            new_source = []
            skip_old_path_block = False
            for line in source:
                # Detect the start of the old CPT validation block
                if f"if not os.path.exists('/kaggle/working/{cpt_dir_name}/phase_1b/config.json'):" in line or f"if not os.path.exists('/kaggle/working/{cpt_dir_name}/phase_1b/config.json\'):" in line:
                    skip_old_path_block = True
                    
                    # Insert the SFT Resumption Fallback logic
                    new_source.append("# Check for SFT checkpoints to enable robust resumption fallback\n")
                    new_source.append("sft_dir = \"/kaggle/working/sft_checkpoints\"\n")
                    new_source.append("sft_checkpoint = None\n")
                    new_source.append("if os.path.exists(sft_dir):\n")
                    new_source.append("    subdirs = [d for d in os.listdir(sft_dir) if d.startswith('checkpoint-')]\n")
                    new_source.append("    if subdirs:\n")
                    new_source.append("        subdirs.sort(key=lambda x: int(x.split('-')[1]), reverse=True)\n")
                    new_source.append("        sft_checkpoint = os.path.join(sft_dir, subdirs[0])\n")
                    new_source.append("        print(f\"Detected restored SFT checkpoint: {sft_checkpoint}\")\n")
                    new_source.append("\n")
                    new_source.append(f"model_path = '{cpt_root}'\n")
                    new_source.append("if not os.path.exists(os.path.join(model_path, 'config.json')):\n")
                    new_source.append("    if os.path.exists(model_path):\n")
                    new_source.append("        subdirs = [d for d in os.listdir(model_path) if d.startswith('checkpoint-')]\n")
                    new_source.append("        if subdirs:\n")
                    new_source.append("            subdirs.sort(key=lambda x: int(x.split('-')[1]), reverse=True)\n")
                    new_source.append("            model_path = os.path.join(model_path, subdirs[0])\n")
                    new_source.append("            print(f\"CPT config.json not found in root, resolved to latest checkpoint: {model_path}\")\n")
                    new_source.append("\n")
                    new_source.append("# Bulletproof check: If CPT weights are missing but SFT checkpoint exists,\n")
                    new_source.append("# we can use the SFT checkpoint as our model_path base to enable safe resumption!\n")
                    new_source.append("has_cpt_weights = os.path.exists(os.path.join(model_path, 'model.safetensors')) or os.path.exists(os.path.join(model_path, 'pytorch_model.bin'))\n")
                    new_source.append("if not has_cpt_weights and sft_checkpoint:\n")
                    new_source.append("    print(f\"⚠️ CPT weights not found in {model_path}, but SFT checkpoint is available.\")\n")
                    new_source.append("    print(f\"👉 Resolving model_path to SFT checkpoint to enable safe resumption: {sft_checkpoint}\")\n")
                    new_source.append("    model_path = sft_checkpoint\n")
                    new_source.append("\n")
                    new_source.append("if not os.path.exists(os.path.join(model_path, 'config.json')):\n")
                    new_source.append("    raise ValueError(f\"Model config not found at {model_path}/config.json! \")\n")
                    continue

                if skip_old_path_block:
                    if "raise ValueError" in line and "config.json" in line:
                        skip_old_path_block = False
                    continue

                # Also patch the --model_path argument in the launch command
                if f"--model_path /kaggle/working/{cpt_dir_name}/phase_1b" in line:
                    line = line.replace(f"--model_path /kaggle/working/{cpt_dir_name}/phase_1b", "--model_path {model_path}")

                new_source.append(line)

            cell["source"] = new_source
            updated = True
            print("     ✅ Patched SFT Cell!")

        # 3. Update Step 4 (Inference Cell)
        elif "scripts/test_inference.py" in source_str:
            print("  -> Patching Inference Cell...")
            new_source = []
            skip_old_inference_block = False
            for line in source:
                # Detect SFT config check
                if 'if not os.path.exists(os.path.join(model_path, "config.json")):' in line or "if not os.path.exists(os.path.join(model_path, 'config.json')):" in line:
                    skip_old_inference_block = True
                    
                    # Insert the new robust SFT checkpoint / CPT checkpoint fallback
                    new_source.append("if not os.path.exists(os.path.join(model_path, \"config.json\")):\n")
                    new_source.append("    # Try SFT checkpoint first\n")
                    new_source.append("    sft_subdirs = [d for d in os.listdir(model_path) if d.startswith('checkpoint-')] if os.path.exists(model_path) else []\n")
                    new_source.append("    if sft_subdirs:\n")
                    new_source.append("        sft_subdirs.sort(key=lambda x: int(x.split('-')[1]), reverse=True)\n")
                    new_source.append("        model_path = os.path.join(model_path, sft_subdirs[0])\n")
                    new_source.append("        print(f\"SFT not finished, resolved to latest SFT checkpoint: {model_path}\")\n")
                    new_source.append("    else:\n")
                    new_source.append("        print(\"SFT not finished or not saved, falling back to CPT checkpoint...\")\n")
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
        print(f"❌ No matching cells found/updated in {notebook_path}.\n")
        return False

if __name__ == "__main__":
    notebooks = [
        # Bori-2 notebooks
        "/Users/brandon.baek/Development/Bori/Bori-2/kaggle_runner_v2.ipynb",
        "/Users/brandon.baek/Development/Bori/Bori-2/codebase/kaggle_runner_v2.ipynb",
        # Bori-3 notebooks
        "/Users/brandon.baek/Development/Bori/Bori-3/kaggle_runner_v3.ipynb",
        "/Users/brandon.baek/Development/Bori/Bori-3/codebase/kaggle_runner_v3.ipynb"
    ]
    
    for nb_path in notebooks:
        patch_notebook(nb_path)

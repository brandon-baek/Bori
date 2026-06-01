import json
import os

def fix_tokenizer_restore(notebook_path):
    print(f"Updating restore logic in: {notebook_path}")
    if not os.path.exists(notebook_path):
        print(f"Error: Notebook not found at {notebook_path}")
        return False
        
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    updated = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code" and any("Searching for existing checkpoints and merged tokenizer recursively" in line for line in cell.get("source", [])):
            source = cell["source"]
            
            # Find the starting index of our previous restoration block
            start_idx = -1
            end_idx = -1
            for idx, line in enumerate(source):
                if "# ──── Robust Recursive Checkpoint & Merged Tokenizer Restoration ────" in line:
                    start_idx = idx
                if "print(\"Working directory structure:\")" in line:
                    end_idx = idx
                    break
                    
            if start_idx != -1 and end_idx != -1:
                new_restore_code = [
                    "\n",
                    "# ──── Robust Explicit Checkpoint & Merged Tokenizer Restoration ────\n",
                    "print(\"Restoring checkpoints and merged tokenizer from Kaggle inputs...\")\n",
                    "found_checkpoints = False\n",
                    "found_tokenizer = False\n",
                    "\n",
                    "cpt_dataset_path = \"/kaggle/input/datasets/brandonbaek/bori-v2-checkpoints\"\n",
                    "code_dataset_path = \"/kaggle/input/datasets/brandonbaek/bori-2-code\"\n",
                    "\n",
                    "def extract_zip_to(zip_path, dest_dir):\n",
                    "    import zipfile\n",
                    "    print(f\"Extracting zipped checkpoints {zip_path} to {dest_dir}...\")\n",
                    "    os.makedirs(dest_dir, exist_ok=True)\n",
                    "    with zipfile.ZipFile(zip_path, 'r') as zip_ref:\n",
                    "        zip_ref.extractall(dest_dir)\n",
                    "    print(f\"✅ Extracted successfully to {dest_dir}\")\n",
                    "\n",
                    "if os.path.exists(cpt_dataset_path):\n",
                    "    print(f\"CPT checkpoints dataset found at: {cpt_dataset_path}\")\n",
                    "    \n",
                    "    # Find if there are zip files inside\n",
                    "    zip_files = []\n",
                    "    for root, dirs, files in os.walk(cpt_dataset_path):\n",
                    "        for f in files:\n",
                    "            if f.endswith(\".zip\"):\n",
                    "                zip_files.append(os.path.join(root, f))\n",
                    "                \n",
                    "    if zip_files:\n",
                    "        for z in zip_files:\n",
                    "            z_name = os.path.basename(z).lower()\n",
                    "            if \"cpt\" in z_name or \"phase\" in z_name:\n",
                    "                extract_zip_to(z, \"/kaggle/working/checkpoints\")\n",
                    "                found_checkpoints = True\n",
                    "            elif \"sft\" in z_name:\n",
                    "                extract_zip_to(z, \"/kaggle/working/sft_checkpoints\")\n",
                    "            else:\n",
                    "                extract_zip_to(z, \"/kaggle/working/checkpoints\")\n",
                    "                found_checkpoints = True\n",
                    "    else:\n",
                    "        # Unzipped checkpoints dataset\n",
                    "        print(f\"Copying unzipped checkpoints from {cpt_dataset_path}...\")\n",
                    "        has_phases = any(x in os.listdir(cpt_dataset_path) for x in [\"phase_1a\", \"phase_1b\", \"checkpoint-2\", \"checkpoint-4\"])\n",
                    "        if has_phases:\n",
                    "            shutil.copytree(cpt_dataset_path, \"/kaggle/working/checkpoints\", dirs_exist_ok=True)\n",
                    "            found_checkpoints = True\n",
                    "            print(\"✅ Copied checkpoints to /kaggle/working/checkpoints\")\n",
                    "        else:\n",
                    "            # Check nested subdirectories\n",
                    "            for root, dirs, files in os.walk(cpt_dataset_path):\n",
                    "                if \"phase_1a\" in dirs or \"phase_1b\" in dirs or any(d.startswith(\"checkpoint-\") for d in dirs):\n",
                    "                    print(f\"Found nested checkpoints directory at: {root}\")\n",
                    "                    shutil.copytree(root, \"/kaggle/working/checkpoints\", dirs_exist_ok=True)\n",
                    "                    found_checkpoints = True\n",
                    "                    print(\"✅ Successfully restored CPT checkpoints to /kaggle/working/checkpoints\")\n",
                    "                    break\n",
                    "\n",
                    "# Now restore tokenizer from restored checkpoints or CPT dataset directly\n",
                    "if os.path.exists(\"/kaggle/working/checkpoints\"):\n",
                    "    for root, dirs, files in os.walk(\"/kaggle/working/checkpoints\"):\n",
                    "        if \"tokenizer.json\" in files and \"tokenizer_config.json\" in files:\n",
                    "            print(f\"Restoring tokenizer from CPT checkpoints folder: {root}\")\n",
                    "            dest_tok = \"/kaggle/working/data/merged_tokenizer\"\n",
                    "            os.makedirs(dest_tok, exist_ok=True)\n",
                    "            for f in os.listdir(root):\n",
                    "                if any(x in f for x in [\"tokenizer\", \"vocab\", \"special_tokens\", \"config\", \"added_tokens\"]):\n",
                    "                    shutil.copy2(os.path.join(root, f), os.path.join(dest_tok, f))\n",
                    "            found_tokenizer = True\n",
                    "            print(\"✅ Successfully restored merged tokenizer to data/merged_tokenizer\")\n",
                    "            break\n",
                    "\n",
                    "if not found_tokenizer and os.path.exists(cpt_dataset_path):\n",
                    "    for root, dirs, files in os.walk(cpt_dataset_path):\n",
                    "        if \"tokenizer.json\" in files and \"tokenizer_config.json\" in files:\n",
                    "            print(f\"Restoring tokenizer directly from CPT dataset: {root}\")\n",
                    "            dest_tok = \"/kaggle/working/data/merged_tokenizer\"\n",
                    "            os.makedirs(dest_tok, exist_ok=True)\n",
                    "            for f in os.listdir(root):\n",
                    "                if any(x in f for x in [\"tokenizer\", \"vocab\", \"special_tokens\", \"config\", \"added_tokens\"]):\n",
                    "                    shutil.copy2(os.path.join(root, f), os.path.join(dest_tok, f))\n",
                    "            found_tokenizer = True\n",
                    "            print(\"✅ Successfully restored merged tokenizer to data/merged_tokenizer\")\n",
                    "            break\n",
                    "\n",
                    "if not found_checkpoints:\n",
                    "    print(\"ℹ️ No restored checkpoints found in inputs. Starting CPT from scratch.\")\n",
                    "if not found_tokenizer:\n",
                    "    print(\"ℹ️ No restored merged tokenizer found. Will train one if needed.\")\n",
                    "\n"
                ]
                
                cell["source"] = source[:start_idx] + new_restore_code + source[end_idx:]
                updated = True
                break
                
    if updated:
        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print("✅ Restore block updated successfully!")
        return True
    else:
        print("❌ Could not locate previous restoration block in the notebook.")
        return False

fix_tokenizer_restore("/Users/brandon.baek/Development/Bori/Bori_Version2/kaggle_runner_v2.ipynb")
fix_tokenizer_restore("/Users/brandon.baek/Development/Bori/Bori_Version2/bori_v2/kaggle_runner_v2.ipynb")

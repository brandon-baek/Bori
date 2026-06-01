import json
import os

def update_notebook(notebook_path):
    print(f"Updating notebook: {notebook_path}")
    if not os.path.exists(notebook_path):
        print(f"Error: Notebook not found at {notebook_path}")
        return False
        
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    # Find the Setup Codebase code cell
    updated = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code" and any("Cleaning working directory" in line for line in cell.get("source", [])):
            print("Found Setup Codebase cell!")
            source = cell["source"]
            
            # Locate where we should insert the recursive checkpoints/tokenizer restore block
            # We want to insert it right before the "Working directory structure" printout
            insert_idx = -1
            for idx, line in enumerate(source):
                if 'print("Working directory structure:")' in line or 'print("Working directory structure:' in line:
                    insert_idx = idx
                    break
                    
            if insert_idx != -1:
                restore_code = [
                    "\n",
                    "# ──── Robust Recursive Checkpoint & Merged Tokenizer Restoration ────\n",
                    "print(\"Searching for existing checkpoints and merged tokenizer recursively in Kaggle inputs...\")\n",
                    "found_checkpoints = False\n",
                    "found_tokenizer = False\n",
                    "\n",
                    "for root, dirs, files in os.walk(input_dir):\n",
                    "    # Skip standard directories to avoid unnecessary searching or loading code files\n",
                    "    if any(x in root for x in [\"bori-2-code\", \"bori_v2\"]):\n",
                    "        continue\n",
                    "\n",
                    "    # 1. Look for CPT checkpoints (must contain phase_1a, phase_1b or checkpoint- subdirectories)\n",
                    "    if not found_checkpoints:\n",
                    "        if \"phase_1a\" in dirs or \"phase_1b\" in dirs or any(d.startswith(\"checkpoint-\") for d in dirs):\n",
                    "            print(f\"Found restored checkpoints directory at: {root}\")\n",
                    "            dest_cpts = \"/kaggle/working/checkpoints\"\n",
                    "            os.makedirs(dest_cpts, exist_ok=True)\n",
                    "            for item in os.listdir(root):\n",
                    "                s = os.path.join(root, item)\n",
                    "                d = os.path.join(dest_cpts, item)\n",
                    "                if os.path.isdir(s):\n",
                    "                    shutil.copytree(s, d, dirs_exist_ok=True)\n",
                    "                else:\n",
                    "                    shutil.copy2(s, d)\n",
                    "            found_checkpoints = True\n",
                    "            print(\"✅ Successfully restored CPT checkpoints to /kaggle/working/checkpoints\")\n",
                    "\n",
                    "    # 2. Look for SFT checkpoints\n",
                    "    if \"sft_checkpoints\" in dirs:\n",
                    "        sft_src = os.path.join(root, \"sft_checkpoints\")\n",
                    "        print(f\"Found restored SFT checkpoints at: {sft_src}\")\n",
                    "        shutil.copytree(sft_src, \"/kaggle/working/sft_checkpoints\", dirs_exist_ok=True)\n",
                    "        print(\"✅ Successfully restored SFT checkpoints to /kaggle/working/sft_checkpoints\")\n",
                    "\n",
                    "    # 3. Look for merged tokenizer files\n",
                    "    if not found_tokenizer:\n",
                    "        if \"tokenizer.json\" in files and \"tokenizer_config.json\" in files:\n",
                    "            # Check if this tokenizer path contains 'merged' or is part of checkpoints\n",
                    "            # to make sure it's the custom tokenizer and not a base model cached tokenizer\n",
                    "            if \"merged\" in root.lower() or \"checkpoints\" in root.lower():\n",
                    "                print(f\"Found merged tokenizer files at: {root}\")\n",
                    "                dest_tok = \"/kaggle/working/data/merged_tokenizer\"\n",
                    "                os.makedirs(dest_tok, exist_ok=True)\n",
                    "                for f in os.listdir(root):\n",
                    "                    if any(x in f for x in [\"tokenizer\", \"vocab\", \"special_tokens\", \"config\", \"added_tokens\"]):\n",
                    "                        shutil.copy2(os.path.join(root, f), os.path.join(dest_tok, f))\n",
                    "                found_tokenizer = True\n",
                    "                print(\"✅ Successfully restored merged tokenizer to /kaggle/working/data/merged_tokenizer\")\n",
                    "\n",
                    "if not found_checkpoints:\n",
                    "    print(\"ℹ️ No restored checkpoints found in inputs. Starting CPT from scratch or waiting for training.\")\n",
                    "if not found_tokenizer:\n",
                    "    print(\"ℹ️ No restored merged tokenizer found in inputs. Will train one if needed.\")\n",
                    "\n"
                ]
                
                # Insert our restoration code right before the working directory structure printout
                cell["source"] = source[:insert_idx] + restore_code + source[insert_idx:]
                updated = True
                break
                
    if updated:
        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print("✅ Notebook successfully updated!")
        return True
    else:
        print("❌ Could not find proper insertion point in the notebook.")
        return False

# Update both notebooks
update_notebook("/Users/brandon.baek/Development/Bori/Bori_Version2/kaggle_runner_v2.ipynb")
update_notebook("/Users/brandon.baek/Development/Bori/Bori_Version2/bori_v2/kaggle_runner_v2.ipynb")

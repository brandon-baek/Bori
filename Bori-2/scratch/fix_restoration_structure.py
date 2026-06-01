import json
import os

def fix_restoration_structure(notebook_path):
    print(f"Updating restoration structure in: {notebook_path}")
    if not os.path.exists(notebook_path):
        print(f"Error: Notebook not found at {notebook_path}")
        return False
        
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    updated = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code" and any("Robust Explicit Checkpoint" in line for line in cell.get("source", [])):
            source = cell["source"]
            
            # Find the starting index of our previous restoration block
            start_idx = -1
            end_idx = -1
            for idx, line in enumerate(source):
                if "# ──── Robust Explicit Checkpoint & Merged Tokenizer Restoration ────" in line:
                    start_idx = idx
                if "print(\"Working directory structure:\")" in line:
                    end_idx = idx
                    break
                    
            if start_idx != -1 and end_idx != -1:
                direct_restore_code = [
                    "\n",
                    "# ──── Robust Direct Checkpoint & Merged Tokenizer Restoration ────\n",
                    "print(\"Restoring checkpoints and merged tokenizer directly from Kaggle input dataset...\")\n",
                    "found_checkpoints = False\n",
                    "found_tokenizer = False\n",
                    "\n",
                    "dataset_root = \"/kaggle/input/datasets/brandonbaek/bori-v2-checkpoints\"\n",
                    "checkpoints_src = os.path.join(dataset_root, \"checkpoints\")\n",
                    "tokenizer_src = os.path.join(dataset_root, \"data/merged_tokenizer\")\n",
                    "\n",
                    "# 1. Direct restore of checkpoints folder\n",
                    "if os.path.exists(checkpoints_src):\n",
                    "    print(f\"Found checkpoints directory at: {checkpoints_src}\")\n",
                    "    shutil.copytree(checkpoints_src, \"/kaggle/working/checkpoints\", dirs_exist_ok=True)\n",
                    "    found_checkpoints = True\n",
                    "    print(\"✅ Successfully restored CPT checkpoints to /kaggle/working/checkpoints\")\n",
                    "else:\n",
                    "    # Search fallback if checkpoints folder is nested differently\n",
                    "    for root, dirs, files in os.walk(dataset_root):\n",
                    "        if \"phase_1a\" in dirs or \"phase_1b\" in dirs or any(d.startswith(\"checkpoint-\") for d in dirs):\n",
                    "            # Avoid copying if it's already inside a nested subdirectory we already processed\n",
                    "            if \"working/checkpoints\" in root:\n",
                    "                continue\n",
                    "            print(f\"Found nested checkpoints directory at: {root}\")\n",
                    "            shutil.copytree(root, \"/kaggle/working/checkpoints\", dirs_exist_ok=True)\n",
                    "            found_checkpoints = True\n",
                    "            print(\"✅ Successfully restored CPT checkpoints to /kaggle/working/checkpoints\")\n",
                    "            break\n",
                    "\n",
                    "# 2. Direct restore of merged tokenizer from the data/ folder\n",
                    "if os.path.exists(tokenizer_src):\n",
                    "    print(f\"Found merged tokenizer directory at: {tokenizer_src}\")\n",
                    "    shutil.copytree(tokenizer_src, \"/kaggle/working/data/merged_tokenizer\", dirs_exist_ok=True)\n",
                    "    found_tokenizer = True\n",
                    "    print(\"✅ Successfully restored merged tokenizer to /kaggle/working/data/merged_tokenizer\")\n",
                    "else:\n",
                    "    # Fallback recursive search for tokenizer inside the dataset\n",
                    "    for root, dirs, files in os.walk(dataset_root):\n",
                    "        if \"working/data\" in root:\n",
                    "            continue\n",
                    "        if \"tokenizer.json\" in files and \"tokenizer_config.json\" in files:\n",
                    "            # Match either a merged tokenizer folder or a checkpoints folder\n",
                    "            if \"merged\" in root.lower() or \"checkpoint\" in root.lower() or \"phase_1a\" in root.lower() or \"phase_1b\" in root.lower():\n",
                    "                print(f\"Found merged tokenizer files at: {root}\")\n",
                    "                dest_tok = \"/kaggle/working/data/merged_tokenizer\"\n",
                    "                os.makedirs(dest_tok, exist_ok=True)\n",
                    "                for f in os.listdir(root):\n",
                    "                    if any(x in f for x in [\"tokenizer\", \"vocab\", \"special_tokens\", \"config\", \"added_tokens\"]):\n",
                    "                        shutil.copy2(os.path.join(root, f), os.path.join(dest_tok, f))\n",
                    "                found_tokenizer = True\n",
                    "                print(\"✅ Successfully restored merged tokenizer to /kaggle/working/data/merged_tokenizer\")\n",
                    "                break\n",
                    "\n",
                    "# 3. Restore SFT export checkpoints if present\n",
                    "sft_src = os.path.join(dataset_root, \"sft_checkpoints\")\n",
                    "sft_export_src = os.path.join(dataset_root, \"bori_v2_sft_export\")\n",
                    "if os.path.exists(sft_src):\n",
                    "    print(f\"Found SFT checkpoints folder at: {sft_src}\")\n",
                    "    shutil.copytree(sft_src, \"/kaggle/working/sft_checkpoints\", dirs_exist_ok=True)\n",
                    "    print(\"✅ Successfully restored SFT checkpoints to /kaggle/working/sft_checkpoints\")\n",
                    "elif os.path.exists(sft_export_src):\n",
                    "    print(f\"Found SFT export checkpoints folder at: {sft_export_src}\")\n",
                    "    shutil.copytree(sft_export_src, \"/kaggle/working/sft_checkpoints\", dirs_exist_ok=True)\n",
                    "    print(\"✅ Successfully restored SFT export checkpoints to /kaggle/working/sft_checkpoints\")\n",
                    "\n",
                    "if not found_checkpoints:\n",
                    "    print(\"ℹ️ No CPT checkpoints restored from inputs. Starting CPT from scratch.\")\n",
                    "if not found_tokenizer:\n",
                    "    print(\"ℹ️ No merged tokenizer restored from inputs. Will train one if needed.\")\n",
                    "\n"
                ]
                
                cell["source"] = source[:start_idx] + direct_restore_code + source[end_idx:]
                updated = True
                break
                
    if updated:
        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print("✅ Restoration structure updated successfully!")
        return True
    else:
        print("❌ Could not locate previous restoration block in the notebook.")
        return False

fix_restoration_structure("/Users/brandon.baek/Development/Bori/Bori_Version2/kaggle_runner_v2.ipynb")
fix_restoration_structure("/Users/brandon.baek/Development/Bori/Bori_Version2/bori_v2/kaggle_runner_v2.ipynb")

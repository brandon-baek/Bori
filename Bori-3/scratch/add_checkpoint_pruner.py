import json
import os

def add_checkpoint_pruner(notebook_path):
    print(f"Injecting checkpoint pruner into: {notebook_path}")
    if not os.path.exists(notebook_path):
        print(f"Error: Notebook not found at {notebook_path}")
        return False
        
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    updated = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code" and any("Restoring checkpoints and merged tokenizer directly" in line for line in cell.get("source", [])):
            source = cell["source"]
            
            # Find where we print "No restored checkpoints found" or before "Working directory structure"
            insert_idx = -1
            for idx, line in enumerate(source):
                if "if not found_checkpoints:" in line:
                    insert_idx = idx
                    break
                    
            if insert_idx != -1:
                pruner_code = [
                    "\n",
                    "# ──── Robust Startup Local Disk Space Cleanup (Pruning) ────\n",
                    "def prune_local_checkpoints(directory, max_keep=3):\n",
                    "    if not os.path.exists(directory):\n",
                    "        return\n",
                    "    import re\n",
                    "    checkpoint_dirs = []\n",
                    "    for item in os.listdir(directory):\n",
                    "        if item.startswith(\"checkpoint-\"):\n",
                    "            path = os.path.join(directory, item)\n",
                    "            m = re.match(r\"checkpoint-(\\d+)\", item)\n",
                    "            if m:\n",
                    "                checkpoint_dirs.append((int(m.group(1)), path))\n",
                    "    \n",
                    "    if len(checkpoint_dirs) > max_keep:\n",
                    "        checkpoint_dirs.sort()\n",
                    "        to_delete = checkpoint_dirs[:-max_keep]\n",
                    "        print(f\"Disk Cleanup: Found {len(checkpoint_dirs)} local checkpoints in {directory}.\")\n",
                    "        print(f\"Keeping the latest {max_keep} and deleting older ones to free up space...\")\n",
                    "        for step, path in to_delete:\n",
                    "            print(f\"  Deleting older checkpoint: {path}\")\n",
                    "            try:\n",
                    "                shutil.rmtree(path)\n",
                    "            except Exception as e:\n",
                    "                print(f\"  Warning: Could not delete {path}: {e}\")\n",
                    "        print(\"✅ Local disk space cleanup complete!\")\n",
                    "\n",
                    "# Immediately prune CPT and SFT local directories to free up disk space on startup\n",
                    "prune_local_checkpoints(\"/kaggle/working/checkpoints/phase_1a\", max_keep=2)\n",
                    "prune_local_checkpoints(\"/kaggle/working/checkpoints/phase_1b\", max_keep=3)\n",
                    "prune_local_checkpoints(\"/kaggle/working/sft_checkpoints\", max_keep=3)\n",
                    "\n"
                ]
                
                cell["source"] = source[:insert_idx] + pruner_code + source[insert_idx:]
                updated = True
                print("Successfully injected checkpoint pruner into Setup codebase cell!")
                break
                
    if updated:
        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print("✅ Pruner injection complete!")
        return True
    else:
        print("❌ Could not locate proper insertion point for the pruner.")
        return False

add_checkpoint_pruner("/Users/brandon.baek/Development/Bori/Bori_Version2/kaggle_runner_v2.ipynb")
add_checkpoint_pruner("/Users/brandon.baek/Development/Bori/Bori_Version2/bori_v2/kaggle_runner_v2.ipynb")

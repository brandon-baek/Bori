import json
import os

def fix_codebase_mixup(notebook_path):
    print(f"Patching codebase search in: {notebook_path}")
    if not os.path.exists(notebook_path):
        print(f"Error: Notebook not found at {notebook_path}")
        return False
        
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    updated = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code" and any("Searching for codebase recursively" in line for line in cell.get("source", [])):
            source = cell["source"]
            
            # Find the os.walk loop and insert the checkpoint skip filter
            new_source = []
            for line in source:
                new_source.append(line)
                if "for root, dirs, files in os.walk(input_dir):" in line:
                    new_source.append("    # Skip folders that are part of checkpoints to avoid mixing up codebase and checkpoints\n")
                    new_source.append("    if \"checkpoint\" in root.lower() or \"checkpoints\" in root.lower():\n")
                    new_source.append("        continue\n")
                    
            cell["source"] = new_source
            updated = True
            print("Found and successfully patched codebase recursive search loop!")
            break
            
    if updated:
        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print("✅ Patch successful!")
        return True
    else:
        print("❌ Could not locate the codebase search cell.")
        return False

fix_codebase_mixup("/Users/brandon.baek/Development/Bori/Bori_Version2/kaggle_runner_v2.ipynb")
fix_codebase_mixup("/Users/brandon.baek/Development/Bori/Bori_Version2/bori_v2/kaggle_runner_v2.ipynb")

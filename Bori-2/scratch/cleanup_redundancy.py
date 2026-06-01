import json
import os

def clean_redundancy(notebook_path):
    print(f"Cleaning notebook: {notebook_path}")
    if not os.path.exists(notebook_path):
        print(f"Error: Notebook not found at {notebook_path}")
        return False
        
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    updated = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code" and any("CHECKPOINT_DATASET" in line for line in cell.get("source", [])):
            source = cell["source"]
            
            # Rebuild source cell, filtering out the CHECKPOINT_DATASET check
            new_source = []
            skip = False
            for line in source:
                if "CHECKPOINT_DATASET =" in line:
                    skip = True
                    continue
                if skip:
                    # Skip the block following CHECKPOINT_DATASET
                    if "shutil.copytree(" in line or "dirs_exist_ok=True" in line:
                        skip = False
                    continue
                new_source.append(line)
                
            cell["source"] = new_source
            updated = True
            print("Found and cleaned redundant copy block in Step 0d cell!")
            break
            
    if updated:
        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print("✅ Notebook clean complete!")
        return True
    else:
        print("❌ Redundant copy block not found or already cleaned.")
        return False

clean_redundancy("/Users/brandon.baek/Development/Bori/Bori_Version2/kaggle_runner_v2.ipynb")
clean_redundancy("/Users/brandon.baek/Development/Bori/Bori_Version2/bori_v2/kaggle_runner_v2.ipynb")

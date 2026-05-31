import json
import os

def change_sft_id(notebook_path):
    print(f"Updating W&B SFT run ID in: {notebook_path}")
    if not os.path.exists(notebook_path):
        print(f"Error: Notebook not found at {notebook_path}")
        return False
        
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    updated = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code" and any("WANDB_RUN_ID_SFT" in line for line in cell.get("source", [])):
            source = cell["source"]
            
            new_source = []
            for line in source:
                if "WANDB_RUN_ID_SFT =" in line:
                    new_source.append("    WANDB_RUN_ID_SFT = 'bori-v2-135m-sft-run2'\n")
                    updated = True
                else:
                    new_source.append(line)
                    
            cell["source"] = new_source
            print("Found and successfully updated SFT run ID!")
            break
            
    if updated:
        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print("✅ Run ID update complete!")
        return True
    else:
        print("❌ Could not locate the WANDB_RUN_ID_SFT configuration in this notebook.")
        return False

change_sft_id("/Users/brandon.baek/Development/Bori/Bori_Version2/kaggle_runner_v2.ipynb")
change_sft_id("/Users/brandon.baek/Development/Bori/Bori_Version2/bori_v2/kaggle_runner_v2.ipynb")

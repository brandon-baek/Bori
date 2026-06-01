import json
import os

def update_notebook_v3(notebook_path):
    print(f"Upgrading notebook to V3: {notebook_path}")
    if not os.path.exists(notebook_path):
        print(f"Error: Notebook not found at {notebook_path}")
        return False
        
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            source = cell["source"]
            new_source = []
            for line in source:
                # 1. Replace run configurations and names to v3
                line = line.replace("RUN_NAME = 'bori-v2-135m'", "RUN_NAME = 'bori-v3-135m'")
                line = line.replace("WANDB_RUN_ID_CPT = 'bori-v2-135m-cpt'", "WANDB_RUN_ID_CPT = 'bori-v3-135m-cpt'")
                line = line.replace("WANDB_RUN_ID_SFT = 'bori-v2-135m-sft-v2'", "WANDB_RUN_ID_SFT = 'bori-v3-135m-sft'")
                line = line.replace("bori-v2-135m-sft", "bori-v3-135m-sft")
                line = line.replace("bori_v2_cpt_export", "bori_v3_cpt_export")
                line = line.replace("bori_v2_sft_export", "bori_v3_sft_export")
                
                # 2. Rename checkpoints path to cpt_checkpoints
                line = line.replace("/kaggle/working/checkpoints", "/kaggle/working/cpt_checkpoints")
                line = line.replace('item not in ["checkpoints"', 'item not in ["cpt_checkpoints"')
                
                # 3. Dynamic dataset roots and CPT directory mapping
                if "dataset_root = " in line and "bori-v2-checkpoints" in line:
                    new_source.append("    # Support both v2 and v3 checkpoints input datasets backward-compatibly\n")
                    new_source.append("    dataset_root = \"/kaggle/input/datasets/brandonbaek/bori-v3-checkpoints\" if os.path.exists(\"/kaggle/input/datasets/brandonbaek/bori-v3-checkpoints\") else \"/kaggle/input/datasets/brandonbaek/bori-v2-checkpoints\"\n")
                    continue
                    
                line = line.replace("checkpoints_src = os.path.join(dataset_root, \"checkpoints\")", "checkpoints_src = os.path.join(dataset_root, \"checkpoints\") if os.path.exists(os.path.join(dataset_root, \"checkpoints\")) else os.path.join(dataset_root, \"cpt_checkpoints\")")
                line = line.replace("shutil.copytree(checkpoints_src, \"/kaggle/working/cpt_checkpoints\"", "shutil.copytree(checkpoints_src, \"/kaggle/working/cpt_checkpoints\"")
                line = line.replace("shutil.copytree(root, \"/kaggle/working/cpt_checkpoints\"", "shutil.copytree(root, \"/kaggle/working/cpt_checkpoints\"")
                line = line.replace('prune_local_checkpoints("/kaggle/working/checkpoints/phase_1a"', 'prune_local_checkpoints("/kaggle/working/cpt_checkpoints/phase_1a"')
                line = line.replace('prune_local_checkpoints("/kaggle/working/checkpoints/phase_1b"', 'prune_local_checkpoints("/kaggle/working/cpt_checkpoints/phase_1b"')
                
                # 4. SFT code references
                line = line.replace("scripts/cpt.py", "scripts/cpt.py")
                line = line.replace("scripts/sft.py", "scripts/sft.py")
                line = line.replace("bori_v2", "bori_v3")
                
                # 5. CPT command output directories
                line = line.replace("--output_dir /kaggle/working/checkpoints", "--output_dir /kaggle/working/cpt_checkpoints")
                
                new_source.append(line)
            cell["source"] = new_source
            
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"✅ Notebook {notebook_path} successfully upgraded to V3!")
    return True

# Update both notebooks in V3
update_notebook_v3("/Users/brandon.baek/Development/Bori/Bori_Version3/kaggle_runner_v3.ipynb")
update_notebook_v3("/Users/brandon.baek/Development/Bori/Bori_Version3/bori_v3/kaggle_runner_v3.ipynb")

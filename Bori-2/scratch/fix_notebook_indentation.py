import json
import os

def fix_notebook_indentation(filepath, run_id_substring, correct_line):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return False
        
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    modified = False
    for cell in data.get("cells", []):
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])
            for idx, line in enumerate(source):
                if run_id_substring in line:
                    print(f"Found line in {os.path.basename(filepath)}: {repr(line)}")
                    source[idx] = correct_line
                    modified = True
                    print(f"Corrected to: {repr(correct_line)}")
                    
    if modified:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1, ensure_ascii=False)
            f.write("\n")
        print(f"✅ Successfully updated and saved: {filepath}\n")
        return True
    else:
        print(f"⚠️ Warning: No matching line found in {filepath}\n")
        return False

# 1. Fix Bori V2 Notebooks
# The correct line should be just the python code followed by a newline, with NO extra leading spaces or nested quotes
fix_notebook_indentation(
    "/Users/brandon.baek/Development/Bori/Bori_Version2/kaggle_runner_v2.ipynb",
    "WANDB_RUN_ID_SFT =",
    "WANDB_RUN_ID_SFT = 'bori-v2-135m-sft-v2'\n"
)
fix_notebook_indentation(
    "/Users/brandon.baek/Development/Bori/Bori_Version2/bori_v2/kaggle_runner_v2.ipynb",
    "WANDB_RUN_ID_SFT =",
    "WANDB_RUN_ID_SFT = 'bori-v2-135m-sft-v2'\n"
)

# 2. Fix Bori V3 Notebooks
fix_notebook_indentation(
    "/Users/brandon.baek/Development/Bori/Bori_Version3/kaggle_runner_v3.ipynb",
    "WANDB_RUN_ID_SFT =",
    "WANDB_RUN_ID_SFT = 'bori-v3-135m-sft'\n"
)
fix_notebook_indentation(
    "/Users/brandon.baek/Development/Bori/Bori_Version3/bori_v3/kaggle_runner_v3.ipynb",
    "WANDB_RUN_ID_SFT =",
    "WANDB_RUN_ID_SFT = 'bori-v3-135m-sft'\n"
)

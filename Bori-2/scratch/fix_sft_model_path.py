import json
import os

def fix_notebook(notebook_path):
    print(f"Checking notebook: {notebook_path}")
    if not os.path.exists(notebook_path):
        print(f"⚠️ Warning: Notebook not found at {notebook_path}")
        return False

    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    updated = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        source = cell.get("source", [])
        source_str = "".join(source)

        # 1. Handle SFT training cell
        if "scripts/sft.py" in source_str:
            print("  -> Found SFT cell!")
            # Determine CPT checkpoints folder name from cell content
            cpt_root = None
            if "/kaggle/working/checkpoints/phase_1b" in source_str:
                cpt_root = "/kaggle/working/checkpoints/phase_1b"
            elif "/kaggle/working/cpt_checkpoints/phase_1b" in source_str:
                cpt_root = "/kaggle/working/cpt_checkpoints/phase_1b"

            if cpt_root:
                print(f"     Found CPT root path: {cpt_root}")
                # We will reconstruct the cell lines safely
                new_source = []
                skip_original_check = False
                for line in source:
                    # Skip the old validation logic
                    if f"if not os.path.exists('{cpt_root}/config.json'):" in line:
                        skip_original_check = True
                        # Insert the new dynamic check
                        new_source.append(f"model_path = '{cpt_root}'\n")
                        new_source.append("if not os.path.exists(os.path.join(model_path, 'config.json')):\n")
                        new_source.append("    if os.path.exists(model_path):\n")
                        new_source.append("        subdirs = [d for d in os.listdir(model_path) if d.startswith('checkpoint-')]\n")
                        new_source.append("        if subdirs:\n")
                        new_source.append("            subdirs.sort(key=lambda x: int(x.split('-')[1]), reverse=True)\n")
                        new_source.append("            model_path = os.path.join(model_path, subdirs[0])\n")
                        new_source.append("            print(f\"CPT config.json not found in root, resolved to latest checkpoint: {model_path}\")\n")
                        new_source.append("\n")
                        new_source.append("if not os.path.exists(os.path.join(model_path, 'config.json')):\n")
                        new_source.append("    raise ValueError(f\"CPT output model config not found at {model_path}/config.json! \")\n")
                        continue
                    if skip_original_check:
                        if "raise ValueError" in line and f"{cpt_root}/config.json" in line:
                            skip_original_check = False
                        continue
                    
                    # Replace model path in the launch command
                    if f"--model_path {cpt_root}" in line:
                        line = line.replace(f"--model_path {cpt_root}", "--model_path {model_path}")
                    
                    new_source.append(line)
                
                cell["source"] = new_source
                updated = True
                print("     ✅ SFT cell updated with dynamic path resolution!")

        # 2. Handle Step 4 (Inference Fallback cell)
        elif "scripts/test_inference.py" in source_str:
            print("  -> Found Inference cell!")
            cpt_root = None
            if '"/kaggle/working/checkpoints/phase_1b"' in source_str:
                cpt_root = "/kaggle/working/checkpoints/phase_1b"
            elif '"/kaggle/working/cpt_checkpoints/phase_1b"' in source_str:
                cpt_root = "/kaggle/working/cpt_checkpoints/phase_1b"

            if cpt_root:
                print(f"     Found CPT root path in fallback: {cpt_root}")
                new_source = []
                for line in source:
                    new_source.append(line)
                    # Insert the dynamic check right after fallback path assignment
                    if f'model_path = "{cpt_root}"' in line or f"model_path = '{cpt_root}'" in line:
                        new_source.append("    if not os.path.exists(os.path.join(model_path, 'config.json')):\n")
                        new_source.append("        if os.path.exists(model_path):\n")
                        new_source.append("            subdirs = [d for d in os.listdir(model_path) if d.startswith('checkpoint-')]\n")
                        new_source.append("            if subdirs:\n")
                        new_source.append("                subdirs.sort(key=lambda x: int(x.split('-')[1]), reverse=True)\n")
                        new_source.append("                model_path = os.path.join(model_path, subdirs[0])\n")
                        new_source.append("                print(f\"CPT fallback resolved to latest checkpoint: {model_path}\")\n")
                
                cell["source"] = new_source
                updated = True
                print("     ✅ Inference cell updated with dynamic path resolution fallback!")

    if updated:
        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print(f"🎉 Successfully saved changes to {notebook_path}!\n")
        return True
    else:
        print(f"❌ No matching cell patterns found/updated in {notebook_path}.\n")
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
        fix_notebook(nb_path)

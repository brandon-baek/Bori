import json
import os

def fix_notebook_restoration(notebook_path):
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

        if "def restore_latest_checkpoint_only" in source_str:
            print("  -> Found Restoration Cell!")
            if "ALSO: Copy any root level files" in source_str:
                print("     Already updated! Skipping.")
                continue

            new_source = []
            for line in source:
                new_source.append(line)
                if "shutil.copytree(latest_path, dest_path, dirs_exist_ok=True)" in line:
                    # Determine formatting (e.g., if there's a trailing newline in the matched line)
                    has_newline = line.endswith("\n")
                    
                    new_source.append("        # ALSO: Copy any root level files (e.g. model.safetensors, config.json) directly in src_dir to dest_dir\n" if has_newline else "        # ALSO: Copy any root level files (e.g. model.safetensors, config.json) directly in src_dir to dest_dir")
                    new_source.append("        # to ensure that the dest_dir is a fully loaded final model directory if it was saved.\n")
                    new_source.append("        for item in os.listdir(src_dir):\n")
                    new_source.append("            item_path = os.path.join(src_dir, item)\n")
                    new_source.append("            if os.path.isfile(item_path) and not item.startswith(\".\"):\n")
                    new_source.append("                try:\n")
                    new_source.append("                    shutil.copy2(item_path, os.path.join(dest_dir, item))\n")
                    new_source.append("                except Exception as e:\n")
                    new_source.append("                    print(f\"   -> Warning: Could not copy root file {item}: {e}\")\n")
            
            cell["source"] = new_source
            updated = True
            print("     ✅ Restoration function successfully patched!")

    if updated:
        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print(f"🎉 Successfully saved changes to {notebook_path}!\n")
        return True
    else:
        print(f"❌ No matching cell found/updated in {notebook_path}.\n")
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
        fix_notebook_restoration(nb_path)

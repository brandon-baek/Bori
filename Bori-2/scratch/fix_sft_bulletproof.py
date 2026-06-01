import json
import os

def patch_notebook_setup(notebook_path):
    print(f"Checking notebook: {notebook_path}")
    if not os.path.exists(notebook_path):
        print(f"⚠️ Warning: Notebook not found at {notebook_path}")
        return False

    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Determine phase_1b folder name (cpt_checkpoints for v3, checkpoints for v2)
    is_v3 = "v3" in notebook_path.lower()
    cpt_dir_name = "cpt_checkpoints" if is_v3 else "checkpoints"
    dest_phase_1b = f"/kaggle/working/{cpt_dir_name}/phase_1b"

    updated = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        source = cell.get("source", [])
        source_str = "".join(source)

        if "def restore_latest_checkpoint_only" in source_str:
            print("  -> Found Setup/Restoration Cell!")
            if "Post-Restoration Bulletproof Model Weights Validation & Recovery" in source_str:
                print("     Already updated Setup Cell! Skipping.")
                continue

            # We will append the safety check block at the end of this cell
            recovery_code = [
                "\n",
                "\n",
                "# ──── Post-Restoration Bulletproof Model Weights Validation & Recovery ────\n",
                "import glob\n",
                "print(\"Running post-restoration weights verification...\")\n",
                f"dest_phase_1b = \"{dest_phase_1b}\"\n",
                "if os.path.exists(dest_phase_1b):\n",
                "    has_weights = any(os.path.exists(os.path.join(dest_phase_1b, f)) for f in [\"model.safetensors\", \"pytorch_model.bin\"])\n",
                "    has_config = os.path.exists(os.path.join(dest_phase_1b, \"config.json\"))\n",
                "    \n",
                "    if not has_weights or not has_config:\n",
                "        print(\"⚠️ Warning: restored phase_1b root is missing model weights or config.json. Searching recursively in inputs...\")\n",
                "        found_src_weights = None\n",
                "        found_src_config = None\n",
                "        for root, dirs, files in os.walk(dataset_root):\n",
                "            if \"phase_1b\" in root.lower() and \"working\" not in root:\n",
                "                for f in files:\n",
                "                    if f in [\"model.safetensors\", \"pytorch_model.bin\"] and not found_src_weights:\n",
                "                        found_src_weights = os.path.join(root, f)\n",
                "                    if f == \"config.json\" and not found_src_config:\n",
                "                        found_src_config = os.path.join(root, f)\n",
                "        \n",
                "        if found_src_weights and found_src_config:\n",
                "            print(f\"✅ Found weights at: {found_src_weights}\")\n",
                "            print(f\"✅ Found config at: {found_src_config}\")\n",
                "            shutil.copy2(found_src_weights, os.path.join(dest_phase_1b, os.path.basename(found_src_weights)))\n",
                "            shutil.copy2(found_src_config, os.path.join(dest_phase_1b, \"config.json\"))\n",
                "            \n",
                "            # Copy tokenizer and other config files from the same source folder\n",
                "            src_folder = os.path.dirname(found_src_weights)\n",
                "            for f in os.listdir(src_folder):\n",
                "                if any(x in f for x in [\"tokenizer\", \"vocab\", \"special_tokens\", \"config\", \"added_tokens\"]) and os.path.isfile(os.path.join(src_folder, f)):\n",
                "                    shutil.copy2(os.path.join(src_folder, f), os.path.join(dest_phase_1b, f))\n",
                "            print(\"✅ Successfully recovered and populated final pre-trained CPT model files to phase_1b root!\")\n",
                "        else:\n",
                "            print(\"❌ Error: Could not find CPT model weights or config.json in input dataset recursively!\")\n",
                "else:\n",
                "    print(\"❌ Error: phase_1b target directory does not exist. CPT checkpoints restoration might have failed.\")\n"
            ]

            # Make sure the last line of source has a newline character
            if source and not source[-1].endswith("\n"):
                source[-1] = source[-1] + "\n"

            cell["source"] = source + recovery_code
            updated = True
            print("     ✅ Patched Setup cell successfully with bulletproof recovery block!")

    if updated:
        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print(f"🎉 Successfully saved changes to {notebook_path}!\n")
        return True
    else:
        print(f"❌ No matching Setup cell found/updated in {notebook_path}.\n")
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
        patch_notebook_setup(nb_path)

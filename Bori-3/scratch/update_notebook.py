import json
import os

def main():
    nb_path = "kaggle_runner_v3.ipynb"
    if not os.path.exists(nb_path):
        print(f"❌ Notebook {nb_path} not found.")
        return
        
    print(f"Loading notebook {nb_path}...")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    updated = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])
            for idx, line in enumerate(source):
                if "--learning_rate 2e-5" in line:
                    source[idx] = line.replace("--learning_rate 2e-5", "--learning_rate 2e-4")
                    print(f"✅ Found and updated SFT learning rate in notebook: {repr(source[idx])}")
                    updated = True

    if updated:
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print("💾 Notebook saved successfully!")
    else:
        print("⚠️ No learning rate string found to update in the notebook.")

if __name__ == "__main__":
    main()

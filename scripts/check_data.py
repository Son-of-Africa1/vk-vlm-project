from pathlib import Path

base = Path("data/raw")

for folder in [base / "LLaVA-Instruct-ru", base / "GQA-ru"]:
    print(f"\n=== {folder} ===")
    if folder.exists():
        for item in folder.iterdir():
            print(item.name)
    else:
        print("Folder not found.")
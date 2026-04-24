from pathlib import Path
from huggingface_hub import snapshot_download

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

datasets_to_download = [
    ("deepvk/LLaVA-Instruct-ru", RAW_DIR / "LLaVA-Instruct-ru"),
    ("deepvk/GQA-ru", RAW_DIR / "GQA-ru"),
]

for repo_id, local_path in datasets_to_download:
    local_path.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading {repo_id} into: {local_path}\n")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_path)
    )
    print(f"Finished: {repo_id}")
from pathlib import Path
import argparse
from datasets import load_dataset


def load_parquet_folder(folder: Path):
    files = sorted(str(p) for p in folder.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in: {folder}")
    return load_dataset("parquet", data_files=files, split="train")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_size", type=int, default=2000)
    parser.add_argument("--eval_size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", type=str, default="gqa_small_answer")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    raw_dir = base_dir / "data" / "raw" / "GQA-ru"
    out_dir = base_dir / "data" / "processed" / args.name

    train_instr_dir = raw_dir / "train_balanced_instructions"
    eval_instr_dir = raw_dir / "testdev_balanced_instructions"

    print("Loading instruction shards...")
    train_instr = load_parquet_folder(train_instr_dir)
    eval_instr = load_parquet_folder(eval_instr_dir)

    train_n = min(args.train_size, len(train_instr))
    eval_n = min(args.eval_size, len(eval_instr))

    train_subset = train_instr.shuffle(seed=args.seed).select(range(train_n))
    eval_subset = eval_instr.shuffle(seed=args.seed).select(range(eval_n))

    (out_dir / "train").parent.mkdir(parents=True, exist_ok=True)
    train_subset.save_to_disk(str(out_dir / "train"))
    eval_subset.save_to_disk(str(out_dir / "eval"))

    print(f"Saved train subset: {train_n} rows -> {out_dir / 'train'}")
    print(f"Saved eval subset:  {eval_n} rows -> {out_dir / 'eval'}")


if __name__ == "__main__":
    main()
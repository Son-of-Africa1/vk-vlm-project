from pathlib import Path
import argparse
import json
import re
import torch
from datasets import load_dataset, load_from_disk, Image as HFImage
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel


def load_parquet_folder(folder: Path):
    files = sorted(str(p) for p in folder.glob("*.parquet"))
    ds = load_dataset("parquet", data_files=files, split="train")
    if "image" in ds.column_names:
        ds = ds.cast_column("image", HFImage())
    return ds


def build_id_map(image_dataset):
    ids = image_dataset["id"]
    return {img_id: idx for idx, img_id in enumerate(ids)}


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\sа-яё]", "", s)
    return s.strip()


def extract_answer(decoded_text: str) -> str:
    text = decoded_text.strip()

    markers = [
        "Assistant:",
        "assistant",
        "Ответ:",
        "answer",
    ]
    for marker in markers:
        if marker in text:
            text = text.split(marker)[-1].strip()

    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset_name", type=str, default="gqa_main_answer")
    parser.add_argument("--adapter_dir", type=str, default="outputs/gqa_main_answer_smolvlm_256m")
    parser.add_argument("--base_model_id", type=str, default="HuggingFaceTB/SmolVLM-256M-Instruct")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=30)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_dir = Path(__file__).resolve().parent.parent
    raw_dir = base_dir / "data" / "raw" / "GQA-ru"
    proc_dir = base_dir / "data" / "processed" / args.subset_name
    out_dir = base_dir / "outputs" / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_images = load_parquet_folder(raw_dir / "testdev_balanced_images")
    eval_instr = load_from_disk(str(proc_dir / "eval"))
    image_id_to_idx = build_id_map(eval_images)

    processor = AutoProcessor.from_pretrained(args.base_model_id)
    base_model = AutoModelForImageTextToText.from_pretrained(
        args.base_model_id,
        dtype=torch.float16
    )
    model = PeftModel.from_pretrained(base_model, str(base_dir / args.adapter_dir))
    model = model.to(device)
    model.eval()

    total = min(args.num_samples, len(eval_instr))
    exact = 0
    normalized_exact = 0
    results = []

    for i in range(total):
        row = eval_instr[i]
        image = eval_images[image_id_to_idx[row["imageId"]]]["image"].convert("RGB")
        question = row["question"]
        gold = row["answer"]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"Ответь на вопрос по изображению на русском языке.\nВопрос: {question}"}
                ]
            }
        ]

        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = processor(
            text=[prompt],
            images=[[image]],
            return_tensors="pt",
            padding=True
        ).to(device)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False
            )

        decoded = processor.batch_decode(generated, skip_special_tokens=True)[0]
        pred = extract_answer(decoded)

        if pred.strip() == gold.strip():
            exact += 1

        if normalize_text(pred) == normalize_text(gold):
            normalized_exact += 1

        results.append({
            "index": i,
            "question": question,
            "gold_answer": gold,
            "predicted_answer": pred,
            "raw_output": decoded
        })

        print(f"[{i+1}/{total}] done")

    summary = {
        "num_samples": total,
        "exact_match": exact / total if total else 0,
        "normalized_exact_match": normalized_exact / total if total else 0
    }

    with open(out_dir / "gqa_eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(out_dir / "gqa_eval_predictions.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\nEvaluation summary:")
    print(summary)
    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()
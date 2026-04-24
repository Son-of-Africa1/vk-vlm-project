import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from pathlib import Path
import argparse
from PIL import Image
import torch
from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk, Image as HFImage
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


def load_parquet_folder(folder: Path):
    files = sorted(str(p) for p in folder.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in: {folder}")

    ds = load_dataset("parquet", data_files=files, split="train")

    if "image" in ds.column_names:
        ds = ds.cast_column("image", HFImage())

    return ds


def resize_longest_edge(img: Image.Image, longest_edge: int) -> Image.Image:
    w, h = img.size
    max_side = max(w, h)

    if max_side <= longest_edge:
        return img

    scale = longest_edge / max_side
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    return img.resize((new_w, new_h), Image.BICUBIC)


class GQAVLMDataset(Dataset):
    def __init__(
        self,
        instructions,
        images,
        image_id_to_idx,
        target_field="answer",
        image_max_side=384
    ):
        self.instructions = instructions
        self.images = images
        self.image_id_to_idx = image_id_to_idx
        self.target_field = target_field
        self.image_max_side = image_max_side

        self.valid_indices = []
        for i, row in enumerate(self.instructions):
            image_id = row["imageId"]
            target = row.get(self.target_field) or row.get("answer")
            if image_id in self.image_id_to_idx and isinstance(target, str) and target.strip():
                self.valid_indices.append(i)

        print(f"Usable samples: {len(self.valid_indices)}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        row = self.instructions[self.valid_indices[idx]]
        image_id = row["imageId"]

        image_row = self.images[self.image_id_to_idx[image_id]]
        image = image_row["image"].convert("RGB")
        image = resize_longest_edge(image, self.image_max_side)

        question = row["question"].strip()
        answer = (row.get(self.target_field) or row["answer"]).strip()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": f"Ответь на вопрос по изображению на русском языке.\nВопрос: {question}"
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer}
                ],
            },
        ]

        return {"messages": messages, "images": [image]}


def build_id_map(image_dataset):
    ids = image_dataset["id"]
    return {img_id: idx for idx, img_id in enumerate(ids)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset_name", type=str, default="gqa_pilot_answer")
    parser.add_argument("--model_id", type=str, default="HuggingFaceTB/SmolVLM-256M-Instruct")
    parser.add_argument("--target_field", type=str, default="answer", choices=["answer", "fullAnswer"])
    parser.add_argument("--image_max_side", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--warmup_steps", type=int, default=50)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Training cannot proceed on GPU.")

    base_dir = Path(__file__).resolve().parent.parent
    raw_dir = base_dir / "data" / "raw" / "GQA-ru"
    proc_dir = base_dir / "data" / "processed" / args.subset_name
    output_dir = base_dir / "outputs" / f"{args.subset_name}_smolvlm_256m"

    print("Loading image datasets...")
    train_images = load_parquet_folder(raw_dir / "train_balanced_images")
    eval_images = load_parquet_folder(raw_dir / "testdev_balanced_images")

    print("Building image ID maps...")
    train_image_id_to_idx = build_id_map(train_images)
    eval_image_id_to_idx = build_id_map(eval_images)

    print("Loading prepared instruction subsets...")
    train_instr = load_from_disk(str(proc_dir / "train"))
    eval_instr = load_from_disk(str(proc_dir / "eval"))

    print("Loading processor and model...")
    processor = AutoProcessor.from_pretrained(args.model_id)
    processor.tokenizer.padding_side = "right"

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        dtype=torch.float16,
    )

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    train_dataset = GQAVLMDataset(
        instructions=train_instr,
        images=train_images,
        image_id_to_idx=train_image_id_to_idx,
        target_field=args.target_field,
        image_max_side=args.image_max_side,
    )

    eval_dataset = GQAVLMDataset(
        instructions=eval_instr,
        images=eval_images,
        image_id_to_idx=eval_image_id_to_idx,
        target_field=args.target_field,
        image_max_side=args.image_max_side,
    )

    def collate_fn(examples):
        full_texts = []
        prompt_texts = []
        images = []

        for ex in examples:
            full_text = processor.apply_chat_template(
                ex["messages"],
                tokenize=False,
                add_generation_prompt=False
            ).strip()

            prompt_only = processor.apply_chat_template(
                [ex["messages"][0]],
                tokenize=False,
                add_generation_prompt=True
            ).strip()

            full_texts.append(full_text)
            prompt_texts.append(prompt_only)
            images.append([img.convert("RGB") for img in ex["images"]])

        batch = processor(
            text=full_texts,
            images=images,
            return_tensors="pt",
            padding=True
        )

        prompt_batch = processor(
            text=prompt_texts,
            images=images,
            return_tensors="pt",
            padding=True
        )

        labels = batch["input_ids"].clone()

        # Mask padding tokens
        labels[labels == processor.tokenizer.pad_token_id] = -100

        # Mask image tokens if present
        boi_token = processor.tokenizer.special_tokens_map.get("boi_token")
        if boi_token is not None:
            image_token_id = processor.tokenizer.convert_tokens_to_ids(boi_token)
            labels[labels == image_token_id] = -100

        # Extra image token used by some SmolVLM setups
        labels[labels == 262144] = -100

        # Mask the entire prompt/user part, so loss is computed only on assistant answer
        for i in range(labels.size(0)):
            prompt_len = int(prompt_batch["attention_mask"][i].sum().item())
            labels[i, :prompt_len] = -100

        batch["labels"] = labels
        return batch

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,

        eval_strategy="steps",
        eval_steps=args.eval_steps,

        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        optim="adamw_torch",

        report_to="tensorboard",
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        max_length=None,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        processing_class=processor,
        peft_config=peft_config,
    )

    print("Starting training...")
    trainer.train()

    print("Saving model and processor...")
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))

    print(f"Done. Saved to: {output_dir}")


if __name__ == "__main__":
    main()
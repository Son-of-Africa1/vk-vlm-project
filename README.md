@"
# VK Vision-Language Modeling Project

## Overview

This project was completed as part of the VK practical task on Vision-Language Modeling. The aim was to train and evaluate a visual question answering model using VK open data.

The model receives an image and a Russian-language question about that image, then generates an answer in Russian.

## Training Process

The model was not trained from scratch. I used fine-tuning of a pre-trained vision-language model:

`HuggingFaceTB/SmolVLM-256M-Instruct`

Fine-tuning was done using LoRA (Low-Rank Adaptation). This approach was selected because it allows efficient training on limited local GPU resources.

Training was performed locally in Visual Studio Code using an NVIDIA RTX 4050 GPU.

## Dataset Used

The main dataset used was VK’s open dataset:

`deepvk/GQA-ru`

This dataset is available on Hugging Face and contains Russian image-question-answer pairs for visual question answering.

For the main experiment, I used:

- Training samples: 8,000
- Evaluation samples: 500
- Final tested samples: 50

The full dataset was not uploaded to GitHub because of its large size. The `GQA-ru` dataset is approximately 3.97 GB. GitHub normal uploads are limited to 100 MB per file, and large files require Git LFS. Since the dataset is already publicly available on Hugging Face, this repository includes scripts for downloading and preparing it locally instead of duplicating the full dataset.

## Training Configuration

Main experiment configuration:

- Base model: `HuggingFaceTB/SmolVLM-256M-Instruct`
- Dataset: `deepvk/GQA-ru`
- Fine-tuning method: LoRA
- Epochs: 2
- Batch size: 1
- Gradient accumulation: 8
- Image size: 384
- Optimizer: AdamW
- Hardware: NVIDIA RTX 4050 GPU

## Quality Metrics

The model was evaluated using:

- Exact Match
- Normalized Exact Match

Exact Match checks whether the model answer is exactly the same as the correct answer.

Normalized Exact Match compares answers after basic text normalization.

## Main Result

The main 8,000-sample experiment achieved:

- Exact Match: 0.28
- Normalized Exact Match: 0.28

This result was selected as the main project result.

## Comparative Analysis

I also tested an additional pilot configuration with modified label masking.

Pilot experiment:

- Training samples: 2,000
- Evaluation samples: 200
- Exact Match: 0.20

The pilot experiment performed lower than the main experiment. Therefore, the 8,000-sample LoRA fine-tuning run was treated as the stronger and final result.

## Repository Contents

This repository includes:

- training scripts
- dataset preparation scripts
- testing scripts
- evaluation scripts
- requirements file
- trained outputs archive

The trained outputs are included as:

`outputs_full.zip`

This archive contains the trained model adapter, evaluation outputs, and generated result files.

## Main Scripts

- `scripts/download_datasets.py` — downloads VK datasets from Hugging Face
- `scripts/check_data.py` — checks downloaded dataset folders
- `scripts/check_gpu.py` — checks GPU availability
- `scripts/prepare_gqa_subset.py` — prepares training and evaluation subsets
- `scripts/train_gqa_smolvlm.py` — fine-tunes the model
- `scripts/test_gqa_model.py` — tests the trained model on individual samples
- `scripts/evaluate_gqa_model.py` — evaluates the trained model

## Conclusion

The project demonstrates a complete Vision-Language Modeling pipeline using VK open data. It includes dataset preparation, LoRA fine-tuning, GPU training, evaluation, and comparison between two training configurations.

The main result shows that the fine-tuned model achieved moderate performance, and future improvements may include training on more samples, using a larger model, and improving the prompting or evaluation strategy.
"@ | Out-File -Encoding utf8 README.md
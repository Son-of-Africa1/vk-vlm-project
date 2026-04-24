@"
# VK Vision-Language Modeling Project

## Overview

This project was completed for the VK Vision-Language Modeling project task. The aim was to train and evaluate a visual question answering model using open VK data.

The project uses the `deepvk/GQA-ru` dataset and fine-tunes a lightweight vision-language model to answer Russian-language questions about images.

## Project Goal

The goal was to train a Vision-Language Model on VK open data and evaluate its performance on a visual question answering task.

## Dataset

Dataset used:

- `deepvk/GQA-ru`

The dataset contains image-question-answer pairs in Russian. It was used for training and evaluation.

Main experimental setup:

- Training samples: 8,000
- Evaluation samples: 500
- Final tested samples: 50

## Model

Base model:

- `HuggingFaceTB/SmolVLM-256M-Instruct`

Training method:

- LoRA fine-tuning
- GPU-based training

Hardware:

- NVIDIA RTX 4050 GPU

## Main Result

The 8,000-sample training run was treated as the main experimental result.

Final evaluation on 50 samples:

- Exact Match: 0.28
- Normalized Exact Match: 0.28

An additional pilot experiment with modified label masking was also tested, but it achieved a lower score of 0.20 exact match. Therefore, the 8,000-sample main run was selected as the final result.

## Repository Contents

This GitHub repository includes:

- training scripts
- dataset preparation scripts
- testing and evaluation scripts
- project documentation

Large folders are excluded from GitHub because of file size limits:

- `data/raw/`
- `data/processed/`
- `outputs/`
- `.venv/`
- trained model files

The trained outputs can be provided separately through cloud storage if required.

## Main Scripts

- `scripts/download_datasets.py` — downloads VK datasets from Hugging Face
- `scripts/prepare_gqa_subset.py` — prepares training/evaluation subsets
- `scripts/train_gqa_smolvlm.py` — fine-tunes the model
- `scripts/test_gqa_model.py` — tests the trained model on individual samples
- `scripts/evaluate_gqa_model.py` — evaluates the model on multiple samples
- `scripts/check_gpu.py` — checks GPU availability
- `scripts/check_data.py` — checks downloaded dataset folders

## How the Project Was Implemented

The project was implemented locally in Visual Studio Code using Python. The dataset was downloaded from Hugging Face and prepared into smaller training and evaluation subsets. The model was fine-tuned using LoRA to reduce GPU memory requirements.

Training was completed on an NVIDIA RTX 4050 GPU. The trained model was then evaluated using exact match and normalized exact match metrics.

## Training Configuration

Main run configuration:

- Model: `HuggingFaceTB/SmolVLM-256M-Instruct`
- Dataset: `deepvk/GQA-ru`
- Train subset: 8,000 samples
- Evaluation subset: 500 samples
- Epochs: 2
- Batch size: 1
- Gradient accumulation: 8
- Image size: 384
- Optimizer: AdamW
- Fine-tuning method: LoRA

## Conclusion

The project successfully demonstrates a complete Vision-Language Modeling pipeline using VK open data. The model was trained, evaluated, and documented. The final result shows moderate accuracy and provides a reproducible foundation for future improvement using larger datasets, stronger models, and improved prompting strategies.
"@ | Out-File -Encoding utf8 README.md
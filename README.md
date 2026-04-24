@"
# VK Vision-Language Modeling Project

## Overview

This project was completed as part of a VK practical task on Vision-Language Modeling (VLM). The objective was to train and evaluate a multimodal model capable of answering Russian-language questions based on image input.

The project replicates a real-world machine learning workflow using VK open datasets and demonstrates practical implementation of a visual question answering system.

---

## Project Goal

The goal of this project is to fine-tune a Vision-Language Model using VK open datasets and evaluate its performance on a visual question answering (VQA) task.

---

## Dataset

Dataset used:

- `deepvk/GQA-ru`

The dataset consists of image-question-answer pairs in Russian and is designed for VQA tasks.

Main experimental setup:

- Training samples: 8,000  
- Evaluation samples: 500  
- Final evaluation subset: 50  

---

## Model

Base model:

- `HuggingFaceTB/SmolVLM-256M-Instruct`

Training approach:

- LoRA (Low-Rank Adaptation) fine-tuning  
- GPU-based training  

Hardware used:

- NVIDIA RTX 4050 GPU  

---

## Training Configuration

Main run configuration:

- Epochs: 2  
- Batch size: 1  
- Gradient accumulation: 8  
- Image size: 384  
- Optimizer: AdamW  
- Fine-tuning method: LoRA  

---

## Results

Main experiment (8,000 samples):

- Exact Match: 0.28  
- Normalized Exact Match: 0.28  

Additional experiment (pilot, 2,000 samples):

- Exact Match: 0.20  

The 8,000-sample run produced the best performance and is considered the final result.

---

## Repository Contents

This repository includes:

- training scripts  
- dataset preparation scripts  
- evaluation and testing scripts  
- project documentation  
- trained model outputs archive  

---

## Trained Model Outputs

The trained model outputs are included in this repository as:

`outputs_full.zip`

This archive contains:

- trained model adapter  
- evaluation outputs  
- generated prediction results  

---

## Main Scripts

- `scripts/download_datasets.py` — downloads VK datasets  
- `scripts/prepare_gqa_subset.py` — prepares subsets  
- `scripts/train_gqa_smolvlm.py` — model training  
- `scripts/test_gqa_model.py` — sample testing  
- `scripts/evaluate_gqa_model.py` — evaluation  
- `scripts/check_gpu.py` — GPU check  
- `scripts/check_data.py` — dataset validation  

---

## Implementation Summary

The project was implemented in Visual Studio Code using Python. The dataset was downloaded from Hugging Face, processed into subsets, and used to fine-tune a pre-trained multimodal model.

Training was performed on a local GPU using LoRA to reduce memory usage. The model was evaluated using exact match metrics.

---

## Conclusion

The project successfully demonstrates a full Vision-Language Modeling pipeline using VK open data.

Key outcomes:

- functional training pipeline  
- successful GPU-based fine-tuning  
- measurable evaluation results  
- reproducible workflow  

The model achieves moderate accuracy and provides a strong foundation for future improvements using larger datasets, more powerful models, and improved training strategies.

"@ | Out-File -Encoding utf8 README.md
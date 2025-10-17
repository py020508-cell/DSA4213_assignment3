# DSA4213-NLP-Assignment3: Fine-Tuning DistilBERT for IMDb Sentiment Classification

[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.30+-yellow)](https://huggingface.co/docs/transformers)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> A systematic comparison between **full-parameter fine-tuning** and **LoRA efficient fine-tuning** on the IMDb binary sentiment classification task.
> ## 📌 Background
This assignment uses the [IMDb 50k movie review dataset](https://huggingface.co/datasets/imdb) and the lightweight pre-trained language model **DistilBERT** (`distilbert-base-uncased`) to compare two fine-tuning strategies:

| Strategy         | Trainable Params | F1     | Accuracy |
| ---------------- | ---------------- | ------ | -------- |
| Full Fine-Tuning | ~66 M            | 92.3 % | 93.2 %   |
| LoRA (r=8, α=32) | ~0.15 M          | 92.3 % | 93.2 %   |


> Single RTX 3060 Laptop (6 GB), batch_size=16, 3 epochs.

## 🚀 Quick Start

1. **Clone & enter repo**
   ```bash
   git clone https://github.com/py020508-cell/dsa4213-nlp-assignment3.git
   cd dsa4213-nlp-assignment3

2. **Install dependencies**
pip install -r requirements.txt
3. **One-click training**
   -Windows:
     run.bat
   -Linux/Mac:
     chmod +x run.sh && ./run.sh
4. **Standalone evaluation**
python src/model_evaluate.py --ckpt result/full/checkpoint-xxx
python src/model_evaluate.py --ckpt result/lora/checkpoint-xxx

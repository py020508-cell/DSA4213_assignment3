#!/usr/bin/env bash
set -e
echo ">>> 安装依赖"
pip install -r requirements.txt -q
echo ">>> 全参数微调"
python -m src.full_train
echo ">>> LoRA 微调"
python -m src.lora_train
echo ">>> 评估 & 画图"
python -m src.model_evaluate
python -m src.time_evaluate
python -m src.error_analysis
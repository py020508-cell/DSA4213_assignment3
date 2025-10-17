@echo off
echo >>> Installing dependencies
pip install -r requirements.txt -q

echo >>> Full-parameter fine-tuning
python -m src.full_train

echo >>> LoRA fine-tuning
python -m src.lora_train

echo >>> Evaluation & plotting
python -m src.model_evaluate
python -m src.error_analysis

echo >>> Done! Results are saved in the result/ directory.

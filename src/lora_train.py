import os
import json
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, TaskType
from data import TOKENIZER, build_tokenized
from full_train import comp_metrics

def main():
    train, test = build_tokenized()
    base = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8, lora_alpha=32, lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"],
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()


    out_root = "../result/LORA"
    os.makedirs(out_root, exist_ok=True)

    args = TrainingArguments(
        output_dir=out_root,           
        learning_rate=2e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_dir=os.path.join(out_root, "logs")
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=test,
        tokenizer=TOKENIZER,
        data_collator=DataCollatorWithPadding(tokenizer=TOKENIZER),
        compute_metrics=comp_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)


    with open(os.path.join(out_root, "lora_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # save as：../result/checkpoint
    trainer.save_model(os.path.join(out_root, "checkpoint"))

if __name__ == "__main__":
    main()

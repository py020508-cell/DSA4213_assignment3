import os
import json
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from data import TOKENIZER, build_tokenized
import evaluate

# ---------- 工具 ----------
def comp_metrics(eval_pred):
    acc = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = acc.compute(predictions=preds, references=labels)["accuracy"]
    f1 = f1.compute(predictions=preds, references=labels)["f1"]
    return {"accuracy": accuracy, "f1": f1}

# ---------- 训练 ----------
def main():
    train, test = build_tokenized()

    # 1. 输出根目录：../result
    out_root = "../result/full"
    os.makedirs(out_root, exist_ok=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )

    args = TrainingArguments(
        output_dir=out_root,              # 模型、checkpoint、日志全放这里
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir=os.path.join(out_root, "logs"),
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=test,
        tokenizer=TOKENIZER,
        data_collator=DataCollatorWithPadding(tokenizer=TOKENIZER),
        compute_metrics=comp_metrics
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)

    # 2. 指标统一落盘：../result/full_metrics.json
    with open(os.path.join(out_root, "full_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # 3. 最终最佳模型保存：../result/checkpoint
    trainer.save_model(os.path.join(out_root, "checkpoint"))

if __name__ == "__main__":
    main()
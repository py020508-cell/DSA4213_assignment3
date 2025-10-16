# error_analysis_both.py
import json, torch, numpy as np
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, Trainer, DataCollatorWithPadding
from peft import PeftModel
from data import TOKENIZER, build_tokenized   # 你的 tokenizer 与 tokenize 函数

def load_full_model():
    return AutoModelForSequenceClassification.from_pretrained("../result/full/checkpoint")

def load_lora_model():
    base = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    return PeftModel.from_pretrained(base, "../result/LORA/checkpoint")

def predict(model, dataset):
    trainer = Trainer(model=model,
                      data_collator=DataCollatorWithPadding(TOKENIZER),
                      tokenizer=TOKENIZER)
    pred = trainer.predict(dataset)
    logits = pred.predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    preds = logits.argmax(-1)
    return preds, probs

def save_cases(ids, prefix, labels, preds1, probs1, preds2, probs2, dataset):
    out = []
    for i in ids[:3]:
        out.append({
            "id": int(i),
            "text": dataset[i]["text"][:400],
            "true_label": int(labels[i]),
            "full_pred": int(preds1[i]),
            "lora_pred": int(preds2[i]),
            "full_prob": float(probs1[i].max()),
            "lora_prob": float(probs2[i].max())
        })
    with open(f"{prefix}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

def main():
    _, test_ds = build_tokenized()
    labels = np.array(test_ds["label"])

    full_preds, full_probs = predict(load_full_model(), test_ds)
    lora_preds, lora_probs = predict(load_lora_model(), test_ds)

    # ① LoRA 错 & Full 对
    mask_lw = (lora_preds != labels) & (full_preds == labels) & (full_probs.max(1) > 0.85)
    # ② Full 错 & LoRA 对
    mask_fw = (full_preds != labels) & (lora_preds == labels) & (lora_probs.max(1) > 0.85)

    save_cases(np.where(mask_lw)[0], "error_lora_wrong", labels, full_preds, full_probs, lora_preds, lora_probs, test_ds)
    save_cases(np.where(mask_fw)[0], "error_full_wrong", labels, full_preds, full_probs, lora_preds, lora_probs, test_ds)

    print("已导出 error_lora_wrong.json 与 error_full_wrong.json 各 3 条")

if __name__ == "__main__":
    main()
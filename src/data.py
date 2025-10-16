from datasets import load_dataset
from transformers import AutoTokenizer

TOKENIZER = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def get_datasets():
    imdb = load_dataset("imdb")
    train = imdb["train"]
    test  = imdb["test"]
    return train, test

def tokenize(batch):
    return TOKENIZER(batch["text"], truncation=True)

def build_tokenized():
    train, test = get_datasets()
    train = train.map(tokenize, batched=True)
    test  = test.map(tokenize, batched=True)
    return train, test
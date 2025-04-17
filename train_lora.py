import torch
from torch import nn
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments
    )
from datasets import load_dataset
import evaluate
import argparse
from transformer import patch_model_with_lora

# ---------- Argument Parsing ----------
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="sst2", help="GLUE task")
parser.add_argument("--model", type=str, default="roberta-base", help="roberta-base or roberta-large")
parser.add_argument("--r", type=int, default=8)
parser.add_argument("--alpha", type=float, default=None)
parser.add_argument("--output_dir", type=str, default="./results")
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=4e-4)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_seq_length", type=int, default=128)
args = parser.parse_args()

# ---------- Model & Tokenizer ----------
model = RobertaForSequenceClassification.from_pretrained(args.model, num_labels=3 if args.task == "mnli" else 2)
tokenizer = RobertaTokenizer.from_pretrained(args.model)
patch_model_with_lora(model, r=args.r, alpha=args.alpha)

# Freeze all but LoRA weights
for name, param in model.named_parameters():
    param.requires_grad = ("A.weight" in name or "B.weight" in name)

# ---------- Dataset ----------
dataset = load_dataset("glue", args.task)
sentence1_key, sentence2_key = {
    "sst2": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "rte": ("sentence1", "sentence2"),
    "mrpc": ("sentence1", "sentence2"),
    "stsb": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "qnli": ("question", "sentence"),
    "cola": ("sentence", None),
}[args.task]

def tokenize_fn(example):
    return tokenizer(
        example[sentence1_key],
        example[sentence2_key] if sentence2_key else None,
        truncation=True,
        padding="max_length",
        max_length=args.max_seq_length,
    )

encoded_dataset = dataset.map(tokenize_fn, batched=True)
encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# ---------- Metric ----------
metric = evaluate.load("glue", args.task)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=-1)
    return metric.compute(predictions=preds, references=labels)

# ---------- Training ----------
training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=64,
    num_train_epochs=args.epochs,
    learning_rate=args.lr,
    warmup_ratio=0.06,  # from LoRA paper
    eval_strategy="epoch",
    save_strategy="epoch",
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    seed=args.seed,
    lr_scheduler_type="linear",
    max_steps=-1,  # train until epochs are complete
    disable_tqdm=False,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation_matched"] if args.task == "mnli" else encoded_dataset["validation"],
    compute_metrics=compute_metrics,
)

trainer.train()
metrics = trainer.evaluate()
print(metrics)

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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.optim import Adam
import pandas as pd
import os


# ---------- Argument Parsing ----------
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="sst2", help="GLUE task")
parser.add_argument("--model", type=str, default="roberta-base", help="roberta-base or roberta-large")
parser.add_argument("--r", type=int, default=8)
parser.add_argument("--alpha", type=float, default=None)
parser.add_argument("--output_dir", type=str, default="./results")
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_seq_length", type=int, default=128)
parser.add_argument("--warmup_ratio", type=float, default=0.06)
parser.add_argument("--lora_target_weights", nargs="+", default=["query", "value"], help="Which attention weights to apply LoRA to (e.g. query value key dense)")
args = parser.parse_args()

# ---------- Model & Tokenizer ----------
model = RobertaForSequenceClassification.from_pretrained(args.model, num_labels=3 if args.task == "mnli" else 2)
tokenizer = RobertaTokenizer.from_pretrained(args.model)
patch_model_with_lora(model, r=args.r, alpha=args.alpha, target_weights=args.lora_target_weights)

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

def custom_optimizer(model):
    return Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=training_args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08
    )

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
    predictions = torch.argmax(torch.tensor(logits), dim=-1)

    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary" if args.task != "mnli" else "macro")

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# ---------- Training ----------
# AdamW optimizer (used by default by Hugging Face’s Trainer)
training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=64,
    num_train_epochs=args.epochs,
    learning_rate=args.lr, 
    warmup_ratio=args.warmup_ratio,  # from LoRA paper
    eval_strategy="epoch",
    save_strategy="epoch",
    weight_decay=0.1,
    logging_dir="./logs",
    logging_steps=10,
    seed=args.seed,
    lr_scheduler_type="linear", # Linear learning rate decay 
    max_steps=-1,  # train until epochs are complete
    disable_tqdm=False,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation_matched"] if args.task == "mnli" else encoded_dataset["validation"],
    compute_metrics=compute_metrics,
    optimizers=(custom_optimizer(model), None),
)

trainer.train()

# --- Log training results ---
history = trainer.state.log_history
rows = []
for entry in history:
    if "eval_accuracy" in entry:
        rows.append({
            "Training Loss": entry.get("loss", float("nan")),
            "Epoch": entry["epoch"],
            "Step": entry["step"],
            "Validation Loss": entry.get("eval_loss", float("nan")),
            "Accuracy": entry["eval_accuracy"],
            "F1": entry["eval_f1"],
            "Precision": entry["eval_precision"],
            "Recall": entry["eval_recall"]
        })
df = pd.DataFrame(rows)
print("Per-epoch Validation Metrics:")
print(df)
df.to_csv(os.path.join(args.output_dir, "../results/training_results.csv"), index=False)

train_metrics = trainer.evaluate(encoded_dataset["train"])
print("Training Metrics:", train_metrics)
metrics = trainer.evaluate()
print(metrics)

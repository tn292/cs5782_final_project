import torch
from torch import nn
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import evaluate


from transformer import patch_model_with_lora  # your custom patcher

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
patch_model_with_lora(model, r=8, alpha=16)  # apply LoRA to attention layers

# Freeze non-LoRA parameters. freeze original model weights. only trains LoRA
for name, param in model.named_parameters():
    if "A.weight" in name or "B.weight" in name: # only A and B from LoRA adapters are trainable
        param.requires_grad = True
    else:
        param.requires_grad = False

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load and preprocess dataset. SST-2 dataset: binary sentiment classification of single sentences
dataset = load_dataset("glue", "sst2")

#Tokenizes each sentence into BERT input format (input_ids, attention_mask)
def tokenize_fn(example):
    return tokenizer(example["sentence"], truncation=True, padding="max_length", max_length=128)

encoded_dataset = dataset.map(tokenize_fn, batched=True)
encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# accuracy metric
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    return metric.compute(predictions=predictions, references=labels) # compare gainst true labels to return accuracy

# Training config
args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    compute_metrics=compute_metrics,
)

# Train and Evaluate
trainer.train()
trainer.evaluate()

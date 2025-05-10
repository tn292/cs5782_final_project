import os
import subprocess
import pandas as pd
import numpy as np

model = "roberta-base"
task = "mrpc"
r = 8
alpha = 16
lr = 4e-4
batch_size = 16
epochs = 3
max_seq_length = 128
seeds = [42, 43, 44, 45, 46]

for seed in seeds:
    output_dir = f"../results/{task}_r{r}_a{alpha}_s{seed}"
    cmd = [
        "python", "train_lora.py",
        "--task", task,
        "--model", model,
        "--r", str(r),
        "--alpha", str(alpha),
        "--lr", str(lr),
        "--batch_size", str(batch_size),
        "--epochs", str(epochs),
        "--max_seq_length", str(max_seq_length),
        "--seed", str(seed),
        "--output_dir", output_dir
    ]
    print(f"Running {task} with seed {seed} ...")
    subprocess.run(cmd)

# === Compute median best accuracy ===
accuracies = []
for seed in seeds:
    csv_path = f"r../results/{task}_r{r}_a{alpha}_s{seed}/training_results.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        best_acc = df["Accuracy"].max()
        accuracies.append(best_acc)
        print(f"Seed {seed}: best accuracy = {best_acc:.4f}")
    else:
        print(f"Missing results for seed {seed}")

if accuracies:
    median_acc = np.median(accuracies)
    print(f"\nMedian best accuracy over 5 seeds: {median_acc:.4f}")
else:
    print("No results found to compute median.")

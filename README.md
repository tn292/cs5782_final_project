# LoRA Re-Implementation: Low-Rank Adaptation of Large Language Models

## Introduction
This GitHub repository contains a re-implementation of the paper **"[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)"** by Hu et al. (2021). This project was completed as part of a course assignment to explore parameter-efficient fine-tuning strategies. We focus on reproducing the core idea: introducing low-rank trainable updates to frozen pretrained weights.

##  Chosen Result
We reproduced the results presented in **Table 2** of the paper, specifically for the GLUE tasks SST-2, MRPC, and QQP using RoBERTa-base, which highlights that LoRA achieves comparable performance to full fine-tuning using significantly fewer trainable parameters. Our implementation focuses on re-creating the LoRA module for linear layers and testing it on these text classification tasks. 

We also reproduced the results presented in **Table 6** of the paper, specifically for the GLUE task MNLI, which highlights the effect of rank r on model perofmrnace, while adapting the weight type. 

## GitHub Contents
`````
├── lora.py                # LoRA injection logic
├── transformer.py         # RoBERTa model integration with LoRA
├── train_lora.py          # Main training script
├── sweep.py               # 5 seed sweep launcher for getting comparable Table 2 results
├── requirements.txt       # Required dependencies
└── results/               # Folders with output csv files with accuracies, f1, precision and recall scores
`````
## Re-implementation Details
We use Huggingface Transformers with PyTorch to re-implement LoRA on RoBERTa-base. Datasets (SST-2, MRPC, QQP) are loaded from the GLUE benchmark via datasets. Due to compute constraints, all tasks were trained for 3 epochs (vs. 60 in paper for SST-2) and with max sequence length = 128 (vs. 512 in the paper). LoRA injection was done through a custom patch_model_with_lora method.

## Reproduction Steps

1. **Clone the repo**:
    ```bash
    git clone https://github.com/tn292/cs5782_final_project
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the training script** (for example for SST-2 using paper hyperparameters except with our changed max sequence length and number of epochs):
    ```bash
    python train_lora.py --task sst2 --model roberta-base --r 8 --alpha 16 --max_seq_length 128 --batch_size 16 --lr 5e-4 --epochs 3 --seed 42
    ```

**Note**: Requirements: Python 3.10+, PyTorch 2.x, Huggingface Transformers 4.30+, GPU recommended (16GB+ VRAM).

## Results/Insights
We re-implemented LoRA’s results for SST-2, QQP, and MRPC. Compared to the original:

Differences can likely be attributed to shorter training duration and reduced input sequence lengths.

## Conclusion
This re-implementation shows that LoRA can achieve strong performance with fewer parameters, but reproduction fidelity is affected by computational limits. Training time and sequence length significantly impact performance.
## References

- Hu, E., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, L., & Chen, W. (2021). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). *arXiv preprint arXiv:2106.09685*.
- PyTorch Documentation: https://pytorch.org/docs/
- Huggingface Datasets and Transformers libraries

## Acknowledgements
This project was completed as part of the Cornell CS4782/5782: Deep Learning course. We thank the course staff for guidance and support throughout the re-implementation process.
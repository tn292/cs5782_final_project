# LoRA Re-Implementation: Low-Rank Adaptation of Large Language Models

## Introduction
This GitHub repository contains a re-implementation of the paper **"[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)"** by Hu et al. (2021). This project was completed as part of a course assignment to explore parameter-efficient fine-tuning strategies. We focus on reproducing the core idea: introducing low-rank trainable updates to frozen pretrained weights.

##  Chosen Result
We reproduced the results presented in **Table 2** of the paper, specifically for the GLUE tasks SST-2, MRPC, and QQP using RoBERTa-base, which highlights that LoRA achieves comparable performance to full fine-tuning using significantly fewer trainable parameters. Our implementation focuses on re-creating the LoRA module for linear layers and testing it on these text classification tasks. 

We also reproduced the results presented in **Table 6** of the paper, specifically for the GLUE task MNLI, which highlights the effect of rank r on model performance, while adapting the combination of weight types. In our implementation, we used RoBERTa-base instead of GPT-3. 

## GitHub Contents
`````

├── code/lora.py           # LoRA injection logic
├── code/transformer.py    # RoBERTa model integration with LoRA
├── code/train_lora.py     # Main training script
├── code/sweep.py          # 5 seed sweep launcher for getting comparable Table 2 results
├── requirements.txt       # Required dependencies
└── results/               # Folders with output csv files with accuracies, f1, precision and recall scores
`````
## Re-implementation Details
We use Huggingface Transformers with PyTorch to re-implement LoRA on RoBERTa-base. Datasets (SST-2, MRPC, QQP) are loaded from the GLUE benchmark via datasets. Due to compute constraints, all tasks were trained for 3 epochs (vs. 60 in paper for SST-2) and with max sequence length = 128 (vs. 512 in the paper). LoRA injection was done through a custom patch_model_with_lora method. We have also implemented weight type as a training hyperparameter. 

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
    cd code
    python train_lora.py --task mnli --r 8 --alpha 16 --lora_target_weights query value --output_dir ../results/mnli_r8_query_value
    ```

**Note**: Requirements: Python 3.10+, PyTorch 2.x, Huggingface Transformers 4.30+, GPU recommended (16GB+ VRAM).

## Results/Insights
We re-implemented LoRA’s results for SST-2, QQP, and MRPC, replicating Table 2 of the paper. 

Compared to the original:

Differences can likely be attributed to shorter training duration and reduced input sequence lengths. Despite these modifications, our results in table 1 below demonstrate that LoRA can achieve competitive performance with limited training and shorter input lengths. This validates that LoRA is a highly parameter-efficient fine-tuning method capable of strong generalization even under constrained resources. We trained on same hyperparameters used in the paper, outlined in table 2 below.

![Table 1](https://github.com/tn292/cs5782_final_project/blob/main/results/table1.png?raw=true )

![Table 2](https://github.com/tn292/cs5782_final_project/blob/main/results/table2.png?raw=true)

Replicating Table 6, due to evaluating on RoBERTa-base instead of GPT-3, our absolute accuracyes are lower but the trends in performance across ranks and weight types closely follow the paper's original findings. Due to time constraints, we ran each setting across only one random seed, instead of averaging over 5 random seeds, selected the best-performing epoch by validation accuracy, and reported the accuracy. Despite using a smaller model and fewer epochs we confirm the paper’s
key insight: applying LoRA to multiple attention weights improves performance, and increasing the rank, up to a point, generally leads to better accuracy. Results below in table 3. 

![Table 3](https://github.com/tn292/cs5782_final_project/blob/main/results/table3.png?raw=true)

## Conclusion
This re-implementation shows that LoRA can achieve strong performance with fewer parameters, but reproduction fidelity is affected by computational limits. Training time and sequence length significantly impact performance.

Even at 𝑟 = 1, our fine- tuned models performed within ∼ 1% of higher ranks (𝑟 = 8, 16). This minimal gap underscores LoRA’s parameter efficiency and suggests that a fixed low rank often suffices.

Adapting only $W_q, W_v$ yielded almost identical accuracy to adapting all four projections $W_q, W_v, W_k, W_o$. In practice, this means you can save additional parameters by targeting just the most influential attention matrices without sacrificing performance.

Some extensions and future directions include:

- Migrate all experiments from CPU to scalable GPU cloud infrastructure. This would enable reproducing the original epoch counts and exploring larger sequence lengths.
- Implement adaptive rank schedules during
    fine‑tuning similar to ALoRA to allocate more capacity to critical layers, maximizing accuracy per parameter.
- Evaluate LoRA beyond NLP—e.g., vision transformers, speech models, or multimodal tasks—to understand its effectiveness (or limitations) when handling dense inputs like images and/pr audio spectrograms.
## References

- Hu, E., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, L., & Chen, W. (2021). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). *arXiv preprint arXiv:2106.09685*.
- Huggingface Datasets and Transformers libraries
- PyTorch Documentation: https://pytorch.org/docs/

## Acknowledgements
This project was completed as part of the Cornell CS4782/5782: Deep Learning course. We thank the course staff for guidance and support throughout the re-implementation process.

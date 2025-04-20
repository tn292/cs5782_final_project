# LoRA Re-Implementation: Low-Rank Adaptation of Large Language Models

## Introduction
This GitHub repository contains a re-implementation of the paper **"[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)"** by Hu et al. (2021). This project was completed as part of a course assignment to explore parameter-efficient fine-tuning strategies. We focus on reproducing the core idea: introducing low-rank trainable updates to frozen pretrained weights.

##  Chosen Result
We reproduced the results presented in **Table 1** of the paper, which highlights that LoRA achieves comparable performance to full fine-tuning using significantly fewer trainable parameters. Our implementation focuses on re-creating the LoRA module for linear layers and testing it on a text classification task.

## GitHub Contents


## Re-implementation Details
We implemented the LoRA module in PyTorch by decomposing a linear layer's update into two trainable low-rank matrices (A and B), while freezing the original weights. The model was tested on a toy dataset using a small transformer or MLP architecture. Evaluation was based on accuracy and parameter count. Key challenges included weight initialization, stability during training, and integration into existing models.

## Reproduction Steps

1. **Clone the repo**:
    ```bash
    git clone https://github.com/tn292/cs5782_final_project
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Prepare the data**:  
    Place or download the dataset into the `data/` directory.

4. **Run the training script**:
    ```bash

    ```

**Note**: A GPU is recommended for faster training, though CPU should suffice for small models.

## Results/Insights
Our re-implementation confirmed that LoRA achieves performance similar to full fine-tuning while training __ of the parameters. 

## Conclusion

## References

- Hu, E., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, L., & Chen, W. (2021). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). *arXiv preprint arXiv:2106.09685*.
- PyTorch Documentation: https://pytorch.org/docs/


## Acknowledgements
This re-implementation was completed as part of the CS 5782 Deep Learning coursework at Cornell University.
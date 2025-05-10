import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, alpha=16, dropout=0.0, bias = True):
        super().__init__()
        self.r = r
        self.alpha = float(r) if alpha is None else alpha  # Default: alpha = r
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Frozen base linear layer
        self.base = nn.Linear(in_features, out_features, bias = bias)
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        # LoRA parametrized update matrices (trainable)
        self.A = nn.Linear(in_features, r, bias=False)
        self.B = nn.Linear(r, out_features, bias=False)
        self.scaling = self.alpha / self.r

        # Init LoRA weights
        nn.init.normal_(self.A.weight, mean=0.0, std=0.02)  # Gaussian init for A
        nn.init.zeros_(self.B.weight) # zero init for B

    def forward(self, x):
        base_out = self.base(x) # runs the input x through the frozen pre-trained linear layer = W_0 x
        lora_out = self.B(self.A(self.dropout(x))) * self.scaling # = (alpha/r)(BA)x
        return base_out + lora_out # = W_0 x + (BA)x. So h = W_0 x + (alpha/r) (BA)x

import torch.nn as nn
from lora import LoRALinear

def patch_model_with_lora(model, r=8, alpha=16, dropout=0.0):
    """
    Inserts LoRA module into a BERT-like Transformer model
    """
    for name, module in model.named_modules():
        if "encoder.layer" in name and isinstance(module, nn.Module):
            if hasattr(module, "attention"): # limit study to only dapting the attention weights
                attention = module.attention

                # Patch query and value projections. Note: initially chose just query and value bc adapting W_q and W_v yeilds best results as said on pg 10.
                if hasattr(attention.self, "query"): # Adatping query weight. Replace nn.Linear layers with LoRALinear module
                    old = attention.self.query
                    new = LoRALinear(old.in_features, old.out_features, r=r, alpha=alpha, dropout=dropout)
                    new.base.weight.data = old.weight.data.clone()
                    if old.bias is not None:
                        new.base.bias.data = old.bias.data.clone()
                    attention.self.query = new

                if hasattr(attention.self, "value"): # Adapting value weight. Replace nn.Linear layers with LoRALinear module
                    old = attention.self.value
                    new = LoRALinear(old.in_features, old.out_features, r=r, alpha=alpha, dropout=dropout)
                    new.base.weight.data = old.weight.data.clone()
                    if old.bias is not None:
                        new.base.bias.data = old.bias.data.clone()
                    attention.self.value = new

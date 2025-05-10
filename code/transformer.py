import torch.nn as nn
from lora import LoRALinear

def patch_model_with_lora(model, r=8, alpha=16, dropout=0.0, target_weights=("query", "value")):
    for name, module in model.named_modules():
        if "encoder.layer" in name and hasattr(module, "attention"):
            attention = module.attention
            self_attn = attention.self

            if "query" in target_weights and hasattr(self_attn, "query"):
                old = self_attn.query
                new = LoRALinear(old.in_features, old.out_features, r=r, alpha=alpha, dropout=dropout)
                new.base.weight.data = old.weight.data.clone()
                if old.bias is not None:
                    new.base.bias.data = old.bias.data.clone()
                self_attn.query = new

            if "key" in target_weights and hasattr(self_attn, "key"):
                old = self_attn.key
                new = LoRALinear(old.in_features, old.out_features, r=r, alpha=alpha, dropout=dropout)
                new.base.weight.data = old.weight.data.clone()
                if old.bias is not None:
                    new.base.bias.data = old.bias.data.clone()
                self_attn.key = new

            if "value" in target_weights and hasattr(self_attn, "value"):
                old = self_attn.value
                new = LoRALinear(old.in_features, old.out_features, r=r, alpha=alpha, dropout=dropout)
                new.base.weight.data = old.weight.data.clone()
                if old.bias is not None:
                    new.base.bias.data = old.bias.data.clone()
                self_attn.value = new

            if "dense" in target_weights and hasattr(attention.output, "dense"):
                old = attention.output.dense
                new = LoRALinear(old.in_features, old.out_features, r=r, alpha=alpha, dropout=dropout)
                new.base.weight.data = old.weight.data.clone()
                if old.bias is not None:
                    new.base.bias.data = old.bias.data.clone()
                attention.output.dense = new


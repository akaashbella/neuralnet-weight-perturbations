"""
Type: MLP (feedforward).
Plain MLP: no spatial or topological inductive bias.
Supports LayerNorm + Dropout and configurable hidden dims for base/large tiers.
Input: (B, 3, 32, 32) flattened. Output: (B, num_classes).
"""

import torch.nn as nn

INPUT_DIM = 3 * 32 * 32


class MLP(nn.Module):
    def __init__(
        self,
        input_dim=INPUT_DIM,
        hidden_dims=(512, 256),
        num_classes=10,
        dropout=0.0,
        use_layernorm=False,
    ):
        super().__init__()
        self.input_dim = input_dim
        dims = [input_dim] + list(hidden_dims) + [num_classes]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                if use_layernorm:
                    layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.ReLU(inplace=True))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.flatten(1)
        return self.net(x)


# Tier constants for backward compatibility and registry
MLP_SMALL_DIMS = (256, 128)
MLP_MEDIUM_DIMS = (512, 256)
MLP_LARGE_DIMS = (2048, 2048, 1024)
# Base (thesis default): stronger than old medium
MLP_BASE_DIMS = (1024, 1024, 512)

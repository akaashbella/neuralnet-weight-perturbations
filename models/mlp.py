"""
Plain MLP: no spatial or topological inductive bias.
Input: (B, 3, 32, 32) flattened. Output: (B, num_classes).
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim=3 * 32 * 32, hidden_dims=(512, 256), num_classes=10):
        super().__init__()
        self.input_dim = input_dim
        dims = [input_dim] + list(hidden_dims) + [num_classes]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.flatten(1)
        return self.net(x)


# Plan: "Plain MLP" as control. Use medium as canonical; keep small/large for optional use.
MLP_SMALL_DIMS = (256, 128)
MLP_MEDIUM_DIMS = (512, 256)
MLP_LARGE_DIMS = (1024, 512, 256)

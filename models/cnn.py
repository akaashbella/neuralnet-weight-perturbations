"""
CIFAR CNN: 3-stage conv blocks (Conv3x3->BN->ReLU x2 per stage), optional width tier.
Input: (B, 3, 32, 32). Output: (B, num_classes).
- cnn (base): C=64
- cnn_large: C=128
"""

import torch.nn as nn


def _make_stage(in_ch, out_ch, num_blocks=2):
    layers = []
    for i in range(num_blocks):
        layers += [
            nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
    return nn.Sequential(*layers)


class SimpleCNN(nn.Module):
    """
    CIFAR CNN: Stage1 (C ch) -> Pool -> Stage2 (2C) -> Pool -> Stage3 (4C) -> AdaptiveAvgPool -> Dropout -> Linear.
    base_channels C=64 (cnn) or C=128 (cnn_large).
    """

    def __init__(self, num_classes=10, base_channels=64):
        super().__init__()
        C = base_channels
        self.stage1 = _make_stage(3, C)
        self.pool1 = nn.MaxPool2d(2)
        self.stage2 = _make_stage(C, 2 * C)
        self.pool2 = nn.MaxPool2d(2)
        self.stage3 = _make_stage(2 * C, 4 * C)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(4 * C, num_classes),
        )

    def forward(self, x):
        x = self.pool1(self.stage1(x))
        x = self.pool2(self.stage2(x))
        x = self.stage3(x)
        return self.head(x)

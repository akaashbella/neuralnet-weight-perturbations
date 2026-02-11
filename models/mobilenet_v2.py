"""
Type: CNN (MobileNet).
MobileNetV2 with CIFAR-10 stem (first conv stride 1) and configurable width_mult + dropout.
Input: (B, 3, 32, 32). Output: (B, num_classes).
- mobilenet_v2: width_mult=1.0, dropout=0.2
- mobilenet_v2_large: width_mult=1.4, dropout=0.2
"""

import torch.nn as nn
from torchvision.models import mobilenet_v2 as tv_mobilenet_v2


def mobilenet_v2_cifar(num_classes=10, width_mult=1.0, dropout=0.2):
    """MobileNetV2 with CIFAR stem (first conv stride=1) and optional width_mult and dropout."""
    model = tv_mobilenet_v2(
        weights=None,
        num_classes=num_classes,
        width_mult=width_mult,
        dropout=dropout,
    )
    # CIFAR stem: stride=1 so 32×32 doesn't over-downsample
    model.features[0] = nn.Sequential(
        nn.Conv2d(3, model.features[0][0].out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(model.features[0][0].out_channels),
        nn.ReLU6(inplace=True),
    )
    # classifier already has Dropout(dropout) then Linear from torchvision
    return model

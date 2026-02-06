"""
MobileNetV2 with CIFAR-10 stem (first conv stride 1).
Input: (B, 3, 32, 32). Output: (B, num_classes).
"""

import torch.nn as nn
from torchvision.models import mobilenet_v2


def mobilenet_v2_cifar(num_classes=10):
    """MobileNetV2 with CIFAR stem: first conv stride=1 so 32×32 doesn't over-downsample."""
    model = mobilenet_v2(weights=None, num_classes=num_classes)
    model.features[0] = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU6(inplace=True),
    )
    return model

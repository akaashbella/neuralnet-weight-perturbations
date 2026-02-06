"""
Five architectures: MLP (3 variants), Simple CNN, ResNet-18.
All accept 3-channel CIFAR-10 input (B, 3, 32, 32) and output 10-class logits.
MLP variants: mlp_small, mlp_medium, mlp_large (differ by hidden dims).
"""

import torch
import torch.nn as nn
from torchvision import models


# --- MLP: shared base; 3 variants by capacity (small / medium / large) ---

class MLP(nn.Module):
    """MLP: flatten, configurable hidden layers with ReLU, output 10 classes."""

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


# Three MLP variants (same input 3*32*32, 10 classes; differ by depth/width):
# - mlp_small:  (256, 128)     — narrow & shallow, fewer params
# - mlp_medium: (512, 256)     — baseline
# - mlp_large:  (1024, 512, 256) — wider & deeper, more capacity
MLP_SMALL_DIMS = (256, 128)
MLP_MEDIUM_DIMS = (512, 256)
MLP_LARGE_DIMS = (1024, 512, 256)


# --- Simple CNN: Conv2d → ReLU → MaxPool2d x2–3, then FC classifier ---

class SimpleCNN(nn.Module):
    """Conv2d → ReLU → MaxPool2d repeated, then flatten → FC → 10 classes."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # 32 -> 16 -> 8 -> 4 (after three pool-2)
        self.classifier = nn.Linear(64 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


# --- ResNet-18: CIFAR-optimized stem (3×3 stride 1, no maxpool) + final FC for 10 classes ---

def get_resnet18(num_classes=10):
    """
    ResNet-18 with CIFAR-style stem: 3×3 conv stride 1, no maxpool.
    Better suited to 32×32 input than the default ImageNet stem (7×7 stride 2 + maxpool).
    Final FC output num_classes.
    """
    model = models.resnet18(weights=None)
    # CIFAR stem: 3×3 stride 1 instead of 7×7 stride 2; no maxpool
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# --- Factory: single entry point for training/eval so arch names stay consistent ---

def get_model(name, num_classes=10):
    """Return a new model instance by architecture name."""
    if name == "mlp_small":
        return MLP(hidden_dims=MLP_SMALL_DIMS, num_classes=num_classes)
    if name == "mlp_medium":
        return MLP(hidden_dims=MLP_MEDIUM_DIMS, num_classes=num_classes)
    if name == "mlp_large":
        return MLP(hidden_dims=MLP_LARGE_DIMS, num_classes=num_classes)
    if name == "cnn":
        return SimpleCNN(num_classes=num_classes)
    if name == "resnet18":
        return get_resnet18(num_classes=num_classes)
    raise ValueError(f"Unknown architecture: {name}")


ARCH_NAMES = ["mlp_small", "mlp_medium", "mlp_large", "cnn", "resnet18"]


if __name__ == "__main__":
    # Sanity check: each model accepts correct input size and returns (2, 10)
    for name in ARCH_NAMES:
        x = torch.randn(2, 3, 32, 32)
        model = get_model(name)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 10), f"{name}: got {out.shape}"
        print(f"{name}: ok, out.shape={out.shape}")

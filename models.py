"""
Four architectures: MLP, Simple CNN, ResNet-18, ViT.
All accept 3-channel MNIST (B, 3, 28, 28) and output 10-class logits.
ViT resizes 28x28 -> 224x224 internally for patch embedding.
"""

import torch
import torch.nn as nn
from torchvision import models


# --- MLP: 2–3 hidden layers, ReLU, flattened 3*28*28 input ---

class MLP(nn.Module):
    """Simple MLP: flatten, 2–3 hidden layers with ReLU, output 10 classes."""

    def __init__(self, input_dim=3 * 28 * 28, hidden_dims=(512, 256), num_classes=10):
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
        # 28 -> 14 -> 7 -> 3 (after three pool-2)
        self.classifier = nn.Linear(64 * 3 * 3, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


# --- ResNet-18: torchvision, 3-channel OK; replace final FC for 10 classes ---

def get_resnet18(num_classes=10):
    """ResNet-18 with first conv 3-channel (default) and final FC output num_classes."""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# --- ViT: torchvision vit_b_16 (smallest); expects 224×224 input (use get_mnist_loaders(resize=224)) ---

class ViTForMNIST(nn.Module):
    """Torchvision ViT with 10-class head. Expects 224×224 input; use get_mnist_loaders(resize=224) for ViT."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.vit = models.vit_b_16(weights=None, num_classes=num_classes)

    def forward(self, x):
        return self.vit(x)


# --- Factory: single entry point for training/eval so arch names stay consistent ---

def get_model(name, num_classes=10):
    """Return a new model instance by architecture name."""
    if name == "mlp":
        return MLP(num_classes=num_classes)
    if name == "cnn":
        return SimpleCNN(num_classes=num_classes)
    if name == "resnet18":
        return get_resnet18(num_classes=num_classes)
    if name == "vit":
        return ViTForMNIST(num_classes=num_classes)
    raise ValueError(f"Unknown architecture: {name}")


ARCH_NAMES = ["mlp", "cnn", "resnet18", "vit"]


if __name__ == "__main__":
    # Sanity check: each model accepts correct input size and returns (2, 10)
    for name in ARCH_NAMES:
        x = torch.randn(2, 3, 224, 224) if name == "vit" else torch.randn(2, 3, 28, 28)
        model = get_model(name)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 10), f"{name}: got {out.shape}"
        print(f"{name}: ok, out.shape={out.shape}")

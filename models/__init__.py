"""
Model factory and architecture registry.
All models: input (B, 3, 32, 32) [or documented resize], output (B, num_classes).
"""

from .cnn import SimpleCNN
from .mlp import MLP, MLP_BASE_DIMS, MLP_LARGE_DIMS
from .resnet import plainnet20, plainnet56, resnet20, resnet32, resnet56
from .mobilenet_v2 import mobilenet_v2_cifar
from .vit_lite import vit_lite, vit_lite_large
from .gru import GRU
from .lstm import LSTM

# Centralized resize: only architectures that need non-default input size.
# Default is 32 (CIFAR native); list here only if different.
ARCH_INPUT_RESIZE = {}

# Main architectures (one per model file / variant).
ARCH_NAMES = [
    "cnn",
    "mlp",
    "plainnet20",
    "resnet20",
    "mobilenet_v2",
    "vit_lite",
    "gru",
    "lstm",
]

# Large (or deeper) variants of the main architectures.
ARCH_NAMES_LARGE = [
    "cnn_large",
    "mlp_large",
    "mobilenet_v2_large",
    "resnet32",
    "resnet56",
    "plainnet56",
    "vit_lite_large",
    "gru_large",
    "lstm_large",
]

ALL_ARCH_NAMES = ARCH_NAMES + ARCH_NAMES_LARGE


def get_model(name, num_classes=10):
    """Return a new model instance by architecture name."""
    if name == "cnn":
        return SimpleCNN(num_classes=num_classes, base_channels=64)
    if name == "cnn_large":
        return SimpleCNN(num_classes=num_classes, base_channels=128)
    if name == "mlp":
        return MLP(
            hidden_dims=MLP_BASE_DIMS,
            num_classes=num_classes,
            dropout=0.2,
            use_layernorm=True,
        )
    if name == "mlp_large":
        return MLP(
            hidden_dims=MLP_LARGE_DIMS,
            num_classes=num_classes,
            dropout=0.2,
            use_layernorm=True,
        )
    if name == "plainnet20":
        return plainnet20(num_classes=num_classes)
    if name == "plainnet56":
        return plainnet56(num_classes=num_classes)
    if name == "resnet20":
        return resnet20(num_classes=num_classes)
    if name == "resnet32":
        return resnet32(num_classes=num_classes)
    if name == "resnet56":
        return resnet56(num_classes=num_classes)
    if name == "mobilenet_v2":
        return mobilenet_v2_cifar(num_classes=num_classes, width_mult=1.0, dropout=0.2)
    if name == "mobilenet_v2_large":
        return mobilenet_v2_cifar(num_classes=num_classes, width_mult=1.4, dropout=0.2)
    if name == "vit_lite":
        return vit_lite(num_classes=num_classes)
    if name == "vit_lite_large":
        return vit_lite_large(num_classes=num_classes)
    if name == "gru":
        return GRU(num_classes=num_classes)
    if name == "gru_large":
        return GRU(num_classes=num_classes, hidden_size=512, num_layers=2, dropout=0.2)
    if name == "lstm":
        return LSTM(num_classes=num_classes)
    if name == "lstm_large":
        return LSTM(num_classes=num_classes, hidden_size=512, num_layers=2, dropout=0.2)
    raise ValueError(f"Unknown architecture: {name}. Choose from {ALL_ARCH_NAMES}")

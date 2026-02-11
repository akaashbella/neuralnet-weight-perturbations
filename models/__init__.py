"""
Model factory and architecture registry.
All models: input (B, 3, 32, 32) [or documented resize], output (B, num_classes).
"""

from .cnn import SimpleCNN
from .mlp import MLP, MLP_LARGE_DIMS, MLP_MEDIUM_DIMS, MLP_SMALL_DIMS
from .resnet_cifar import plainnet20, resnet20, resnet32
from .mobilenet import mobilenet_v2_cifar
from .vit_lite import vit_lite
from .row_gru import RowGRU
from .row_lstm import RowLSTM

# Centralized resize: only architectures that need non-default input size.
# Default is 32 (CIFAR native); list here only if different.
ARCH_INPUT_RESIZE = {}

# Core set per IMPLEMENTATION_PLAN: floor, control, topology test, residual, efficiency, global.
ARCH_NAMES = [
    "cnn",           # floor
    "mlp",           # control (plain MLP)
    "plainnet20",    # topology test
    "resnet20",      # residual
    "mobilenet_v2",  # efficiency edge case
    "vit_lite",      # global bias
]

# Optional / backward compat (not in core set)
ARCH_NAMES_OPTIONAL = [
    "mlp_small",
    "mlp_medium",
    "mlp_large",
    "resnet32",
    "resnet18",
    "row_gru",
    "row_lstm",
]
ALL_ARCH_NAMES = ARCH_NAMES + ARCH_NAMES_OPTIONAL


def get_model(name, num_classes=10):
    """Return a new model instance by architecture name."""
    if name == "cnn":
        return SimpleCNN(num_classes=num_classes)
    if name == "mlp" or name == "mlp_medium":
        return MLP(hidden_dims=MLP_MEDIUM_DIMS, num_classes=num_classes)
    if name == "mlp_small":
        return MLP(hidden_dims=MLP_SMALL_DIMS, num_classes=num_classes)
    if name == "mlp_large":
        return MLP(hidden_dims=MLP_LARGE_DIMS, num_classes=num_classes)
    if name == "plainnet20":
        return plainnet20(num_classes=num_classes)
    if name == "resnet20":
        return resnet20(num_classes=num_classes)
    if name == "resnet32":
        return resnet32(num_classes=num_classes)
    if name == "mobilenet_v2":
        return mobilenet_v2_cifar(num_classes=num_classes)
    if name == "vit_lite":
        return vit_lite(num_classes=num_classes)
    if name == "row_gru":
        return RowGRU(num_classes=num_classes)
    if name == "row_lstm":
        return RowLSTM(num_classes=num_classes)
    if name == "resnet18":
        from torchvision import models
        import torch.nn as nn
        model = models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f"Unknown architecture: {name}. Choose from {ALL_ARCH_NAMES}")

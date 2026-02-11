"""
Model factory and architecture registry.
All models: input (B, 3, 32, 32) [or documented resize], output (B, num_classes).
"""

from .cnn import SimpleCNN
from .mlp import MLP, MLP_BASE_DIMS, MLP_LARGE_DIMS, MLP_MEDIUM_DIMS, MLP_SMALL_DIMS
from .resnet_cifar import plainnet20, plainnet56, resnet20, resnet32, resnet56
from .mobilenet import mobilenet_v2_cifar
from .vit_lite import vit_lite, vit_lite_large
from .row_gru import RowGRU
from .row_lstm import RowLSTM

# Centralized resize: only architectures that need non-default input size.
# Default is 32 (CIFAR native); list here only if different.
ARCH_INPUT_RESIZE = {}

# Core set (thesis baseline): same training recipe across all.
ARCH_NAMES = [
    "cnn",           # floor
    "mlp",           # control (plain MLP)
    "plainnet20",    # topology test
    "resnet20",      # residual
    "mobilenet_v2",  # efficiency edge case
    "vit_lite",      # global bias
]

# Optional: backward compat + extended capacity (base/large, deeper).
ARCH_NAMES_OPTIONAL = [
    "mlp_small",
    "mlp_small_ln",  # same dims as mlp_small but LayerNorm+dropout (controlled recipe)
    "mlp_medium",
    "mlp_large",
    "cnn_large",
    "mobilenet_v2_large",
    "resnet32",
    "resnet56",
    "plainnet56",
    "vit_lite_large",
    "resnet18",
    "row_gru",
    "row_lstm",
    "row_gru_large",
    "row_lstm_large",
]
ALL_ARCH_NAMES = ARCH_NAMES + ARCH_NAMES_OPTIONAL


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
    if name == "mlp_medium":
        return MLP(
            hidden_dims=MLP_MEDIUM_DIMS,
            num_classes=num_classes,
            dropout=0.2,
            use_layernorm=True,
        )
    if name == "mlp_small":
        # Legacy: no LayerNorm/dropout; use mlp_small_ln for controlled recipe.
        return MLP(hidden_dims=MLP_SMALL_DIMS, num_classes=num_classes)
    if name == "mlp_small_ln":
        return MLP(
            hidden_dims=MLP_SMALL_DIMS,
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
    if name == "row_gru":
        return RowGRU(num_classes=num_classes)
    if name == "row_gru_large":
        return RowGRU(num_classes=num_classes, hidden_size=512, num_layers=2, dropout=0.2)
    if name == "row_lstm":
        return RowLSTM(num_classes=num_classes)
    if name == "row_lstm_large":
        return RowLSTM(num_classes=num_classes, hidden_size=512, num_layers=2, dropout=0.2)
    if name == "resnet18":
        import torch.nn as nn
        from torchvision import models
        model = models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f"Unknown architecture: {name}. Choose from {ALL_ARCH_NAMES}")

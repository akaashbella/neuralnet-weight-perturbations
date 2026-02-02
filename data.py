"""
MNIST data loading with 3-channel conversion for ResNet/ViT compatibility.
Uses standard train/test split; same DataLoader setup for all architectures.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import config


def _to_three_channel(x: torch.Tensor) -> torch.Tensor:
    """Repeat single channel to 3 channels so ResNet/ViT accept MNIST (1,28,28) -> (3,28,28)."""
    return x.repeat(3, 1, 1)


def get_mnist_transform(resize=None):
    """
    Transform: ToTensor, then 1->3 channel. No normalization (0–1 range).
    If resize is given (e.g. 224 for ViT), add Resize so model sees fixed size; avoids resize in forward.
    """
    steps = [
        transforms.ToTensor(),
        transforms.Lambda(_to_three_channel),
    ]
    if resize is not None:
        steps.append(transforms.Resize((resize, resize)))
    return transforms.Compose(steps)


def get_mnist_loaders(batch_size=None, data_dir=None, num_workers=0, resize=None):
    """
    Build train and test DataLoaders for MNIST.
    Uses standard train/test split (train=True/False). Same setup for all architectures.
    resize: if set (e.g. 224 for ViT), images are resized in the transform so the model need not resize in forward.
    """
    batch_size = batch_size or config.BATCH_SIZE
    data_dir = data_dir or config.DATA_DIR
    transform = get_mnist_transform(resize=resize)

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    # If num_workers > 0 later, add generator=torch.Generator().manual_seed(seed) or worker_init_fn for reproducibility.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, test_loader


if __name__ == "__main__":
    # Quick sanity check: load one batch and verify shape
    train_loader, test_loader = get_mnist_loaders()
    x, y = next(iter(train_loader))
    print(f"Train batch: x.shape={x.shape}, y.shape={y.shape}")  # expect (B, 3, 28, 28), (B,)
    x, y = next(iter(test_loader))
    print(f"Test batch:  x.shape={x.shape}, y.shape={y.shape}")

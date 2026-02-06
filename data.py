"""
MNIST and CIFAR-10 data loading. MNIST uses 1->3 channel for ResNet compatibility.
Both datasets: 10 classes; images resized to 28×28 by default (optional resize for other sizes).
"""

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import config


DATASET_CHOICES = ("mnist", "cifar10")


def _to_three_channel(x):
    """Repeat single channel to 3 channels so ResNet accepts MNIST (1,28,28) -> (3,28,28)."""
    return x.repeat(3, 1, 1)


def get_mnist_transform(resize=None):
    """
    Transform: ToTensor, then 1->3 channel. No normalization (0–1 range).
    If resize is given, add Resize so model sees that size (e.g. 224 for ImageNet-sized models).
    """
    steps = [
        transforms.ToTensor(),
        transforms.Lambda(_to_three_channel),
    ]
    if resize is not None:
        steps.append(transforms.Resize((resize, resize)))
    return transforms.Compose(steps)


def get_cifar10_transform(resize=None):
    """
    Transform: ToTensor only (CIFAR-10 is 32×32 RGB). Resize to 28×28 base for model compatibility; optional further resize if given.
    No normalization (0–1 range).
    """
    steps = [transforms.ToTensor()]
    # Base size 28 to match MNIST-sized models (MLP/CNN/ResNet)
    steps.append(transforms.Resize((28, 28)))
    if resize is not None:
        steps.append(transforms.Resize((resize, resize)))
    return transforms.Compose(steps)


def get_mnist_loaders(batch_size=None, data_dir=None, num_workers=0, resize=None):
    """
    Build train and test DataLoaders for MNIST.
    resize: if set, images are resized in the transform (e.g. 224 for larger input).
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


def get_cifar10_loaders(batch_size=None, data_dir=None, num_workers=0, resize=None):
    """
    Build train and test DataLoaders for CIFAR-10.
    Images resized to 28×28 by default (model compatibility); optional resize for other input sizes.
    """
    batch_size = batch_size or config.BATCH_SIZE
    data_dir = data_dir or config.DATA_DIR
    transform = get_cifar10_transform(resize=resize)

    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

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


def get_loaders(dataset, batch_size=None, data_dir=None, num_workers=0, resize=None):
    """
    Build train and test DataLoaders for the given dataset.
    dataset: "mnist" or "cifar10"
    resize: if set, images are resized to that size (e.g. 224 for larger models).
    """
    if dataset == "mnist":
        return get_mnist_loaders(batch_size=batch_size, data_dir=data_dir, num_workers=num_workers, resize=resize)
    if dataset == "cifar10":
        return get_cifar10_loaders(batch_size=batch_size, data_dir=data_dir, num_workers=num_workers, resize=resize)
    raise ValueError(f"Unknown dataset: {dataset}. Choose from {DATASET_CHOICES}")


if __name__ == "__main__":
    for ds in DATASET_CHOICES:
        train_loader, test_loader = get_loaders(ds)
        x, y = next(iter(train_loader))
        print(f"{ds} train batch: x.shape={x.shape}, y.shape={y.shape}")
        x, y = next(iter(test_loader))
        print(f"{ds} test batch:  x.shape={x.shape}, y.shape={y.shape}")

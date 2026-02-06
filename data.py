"""
CIFAR-10 data loading. 32×32 RGB, 10 classes; optional resize for other input sizes.
"""

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import config


DATASET_CHOICES = ("cifar10",)


def get_cifar10_transform(resize=None):
    """
    Transform: ToTensor only (CIFAR-10 is 32×32 RGB). No normalization (0–1 range).
    If resize is given, add Resize so model sees that size (e.g. 224 for ImageNet-sized models).
    """
    steps = [transforms.ToTensor()]
    if resize is not None:
        steps.append(transforms.Resize((resize, resize)))
    return transforms.Compose(steps)


def get_cifar10_loaders(batch_size=None, data_dir=None, num_workers=0, resize=None):
    """
    Build train and test DataLoaders for CIFAR-10.
    Images are 32×32 by default; optional resize for other input sizes.
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


def get_loaders(dataset="cifar10", batch_size=None, data_dir=None, num_workers=0, resize=None):
    """
    Build train and test DataLoaders for the given dataset.
    dataset: "cifar10" (only supported dataset).
    resize: if set, images are resized to that size (e.g. 224 for larger models).
    """
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

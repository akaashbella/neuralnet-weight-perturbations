"""
CIFAR-10 data loading. 32×32 RGB, 10 classes; optional resize for other input sizes.
Uses standard CIFAR-10 normalization and (for training) basic augmentation.
"""

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import config


DATASET_CHOICES = ("cifar10",)

# Standard CIFAR-10 channel-wise mean and std (used for normalization)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_cifar10_transform(train=True, resize=None):
    """
    Resize (if given) is applied first (before ToTensor/Normalize).
    Train: [Resize] + RandomCrop(crop_size, padding=4) + RandomHorizontalFlip + ToTensor + Normalize.
    Test: [Resize] + ToTensor + Normalize.
    crop_size is 32 when resize is None, else resize (so we don't crop to 32 when target is e.g. 224).
    """
    steps = []
    if resize is not None:
        steps.append(transforms.Resize((resize, resize)))

    if train:
        crop_size = resize if resize is not None else 32
        steps += [
            transforms.RandomCrop(crop_size, padding=4),
            transforms.RandomHorizontalFlip(),
        ]

    steps += [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]
    return transforms.Compose(steps)


def get_cifar10_loaders(batch_size=None, data_dir=None, num_workers=0, resize=None):
    """
    Build train and test DataLoaders for CIFAR-10.
    Train uses augmentation (random crop + flip) and normalization; test uses normalization only.
    Images are 32×32 by default; optional resize for other input sizes.
    """
    batch_size = batch_size or config.BATCH_SIZE
    data_dir = data_dir or config.DATA_DIR
    train_transform = get_cifar10_transform(train=True, resize=resize)
    test_transform = get_cifar10_transform(train=False, resize=resize)

    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,
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

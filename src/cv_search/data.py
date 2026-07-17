"""Dataset factories and immutable study splits."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms

from .exceptions import ConfigurationError
from .registry import register_dataset
from .types import DatasetBundle, DatasetMetadata


class TensorImageDataset(Dataset[tuple[torch.Tensor, int]]):
    """Small deterministic tensor dataset used by tests and smoke runs."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        if len(images) != len(labels):
            raise ValueError("images and labels must have equal length")
        self.images = images
        self.labels = labels.long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.images[index], int(self.labels[index])


def fake_classification(config: dict[str, Any]) -> DatasetBundle:
    seed = int(config.get("split_seed", 42))
    generator = torch.Generator().manual_seed(seed)
    image_size = int(config.get("image_size", 32))
    channels = int(config.get("channels", 3))
    classes = int(config.get("num_classes", 10))
    train_size = int(config.get("train_examples", 96))
    validation_size = int(config.get("validation_examples", 32))
    test_size = int(config.get("test_examples", 32))

    def create(size: int) -> TensorImageDataset:
        images = torch.randn(size, channels, image_size, image_size, generator=generator)
        labels = torch.randint(0, classes, (size,), generator=generator)
        return TensorImageDataset(images, labels)

    return DatasetBundle(
        train=create(train_size),
        validation=create(validation_size),
        test=create(test_size),
        metadata=DatasetMetadata(
            name="fake",
            num_classes=classes,
            image_size=(image_size, image_size),
            channels=channels,
            train_examples=train_size,
            validation_examples=validation_size,
            test_examples=test_size,
        ),
    )


def _cifar_transforms(config: dict[str, Any]) -> tuple[Any, Any]:
    augmentation = str(config.get("augmentation", "basic"))
    normalize = transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    )
    evaluation = transforms.Compose([transforms.ToTensor(), normalize])
    if augmentation == "none":
        return evaluation, evaluation
    if augmentation not in {"basic", "strong"}:
        raise ConfigurationError("dataset.augmentation must be none, basic, or strong")
    train_ops: list[Any] = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    if augmentation == "strong":
        train_ops.extend(
            [
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.ToTensor(),
                normalize,
                transforms.RandomErasing(p=0.25),
            ]
        )
    else:
        train_ops.extend([transforms.ToTensor(), normalize])
    return transforms.Compose(train_ops), evaluation


def cifar10(config: dict[str, Any]) -> DatasetBundle:
    root = str(config.get("root", "data"))
    download = bool(config.get("download", False))
    validation_fraction = float(config.get("validation_fraction", 0.1))
    if not 0 < validation_fraction < 1:
        raise ConfigurationError("dataset.validation_fraction must be between 0 and 1")
    split_seed = int(config.get("split_seed", 42))
    train_transform, eval_transform = _cifar_transforms(config)
    full_train_aug = datasets.CIFAR10(
        root=root, train=True, transform=train_transform, download=download
    )
    full_train_eval = datasets.CIFAR10(
        root=root, train=True, transform=eval_transform, download=download
    )
    test = datasets.CIFAR10(root=root, train=False, transform=eval_transform, download=download)
    validation_size = int(round(len(full_train_aug) * validation_fraction))
    train_size = len(full_train_aug) - validation_size
    generator = torch.Generator().manual_seed(split_seed)
    indices = torch.randperm(len(full_train_aug), generator=generator).tolist()
    train_indices, validation_indices = indices[:train_size], indices[train_size:]
    train = Subset(full_train_aug, train_indices)
    validation = Subset(full_train_eval, validation_indices)
    return DatasetBundle(
        train=train,
        validation=validation,
        test=test,
        metadata=DatasetMetadata(
            name="cifar10",
            num_classes=10,
            image_size=(32, 32),
            channels=3,
            train_examples=train_size,
            validation_examples=validation_size,
            test_examples=len(test),
        ),
    )


def fraction_subset(dataset: Dataset[Any], fraction: float, seed: int) -> Dataset[Any]:
    if fraction >= 1.0:
        return dataset
    if not 0 < fraction <= 1:
        raise ConfigurationError("data fraction must be in (0, 1]")
    size = max(1, int(round(len(dataset) * fraction)))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:size].tolist()
    return Subset(dataset, indices)


def register_builtin_datasets() -> None:
    register_dataset("fake", fake_classification, replace=True)
    register_dataset("cifar10", cifar10, replace=True)


register_builtin_datasets()

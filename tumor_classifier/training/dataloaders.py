"""PyTorch DataLoader helpers for image and feature datasets."""

from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

from tumor_classifier.config import DataPaths


def _to_tensor_transform():
    return transforms.Compose([transforms.ToTensor()])


def get_training_loader(batch_size: int, paths: DataPaths) -> DataLoader:
    dataset = torchvision.datasets.ImageFolder(
        root=str(paths.augmented),
        transform=_to_tensor_transform(),
    )
    print("Total training images:", len(dataset))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_validation_loader(batch_size: int, paths: DataPaths) -> DataLoader:
    dataset = torchvision.datasets.ImageFolder(
        root=str(paths.dataset_split / "val"),
        transform=_to_tensor_transform(),
    )
    print("Total validation images:", len(dataset))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def get_test_loader(batch_size: int, paths: DataPaths) -> DataLoader:
    dataset = torchvision.datasets.ImageFolder(
        root=str(paths.dataset_split / "test"),
        transform=_to_tensor_transform(),
    )
    print("Total test images:", len(dataset))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def get_data_loaders(batch_size: int, paths: DataPaths):
    train_loader = get_training_loader(batch_size, paths)
    validation_loader = get_validation_loader(batch_size, paths)
    test_loader = get_test_loader(batch_size, paths)
    classes = train_loader.dataset.classes
    print(classes)
    return train_loader, validation_loader, test_loader, classes


def save_features(
    save_path: Path,
    train_features,
    val_features,
    test_features,
    train_labels,
    val_labels,
    test_labels,
) -> None:
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(train_features, save_path / "train_features.pt")
    torch.save(val_features, save_path / "val_features.pt")
    torch.save(test_features, save_path / "test_features.pt")
    torch.save(train_labels, save_path / "train_labels.pt")
    torch.save(val_labels, save_path / "val_labels.pt")
    torch.save(test_labels, save_path / "test_labels.pt")


def load_saved_features(save_path: Path, batch_size: int = 32):
    train_features = torch.load(save_path / "train_features.pt")
    train_labels = torch.load(save_path / "train_labels.pt")
    val_features = torch.load(save_path / "val_features.pt")
    val_labels = torch.load(save_path / "val_labels.pt")
    test_features = torch.load(save_path / "test_features.pt")
    test_labels = torch.load(save_path / "test_labels.pt")
    return get_feature_loaders(
        train_features,
        train_labels,
        val_features,
        val_labels,
        test_features,
        test_labels,
        batch_size=batch_size,
    )


def get_feature_loaders(
    train_features,
    train_labels,
    val_features,
    val_labels,
    test_features,
    test_labels,
    batch_size: int = 32,
):
    train_loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_features, val_labels),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(test_features, test_labels),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader, test_loader

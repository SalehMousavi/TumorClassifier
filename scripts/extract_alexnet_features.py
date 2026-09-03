#!/usr/bin/env python3
"""Extract AlexNet feature maps for train, validation, and test splits."""

import argparse

from tumor_classifier.config import DataPaths
from tumor_classifier.training.dataloaders import get_data_loaders, save_features
from tumor_classifier.training.feature_extraction import (
    compute_alexnet_features,
    load_alexnet,
)
from tumor_classifier.utils.device import get_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    device = get_device()
    print(f"Using {device}")

    paths = DataPaths()
    train_loader, val_loader, test_loader, _ = get_data_loaders(args.batch_size, paths)
    alexnet = load_alexnet(device)

    train_features, train_labels = compute_alexnet_features(train_loader, alexnet, device)
    val_features, val_labels = compute_alexnet_features(val_loader, alexnet, device)
    test_features, test_labels = compute_alexnet_features(test_loader, alexnet, device)

    save_features(
        paths.feature_maps,
        train_features,
        val_features,
        test_features,
        train_labels,
        val_labels,
        test_labels,
    )
    print(f"Saved feature maps to {paths.feature_maps}")


if __name__ == "__main__":
    main()

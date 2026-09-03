#!/usr/bin/env python3
"""Train the AlexNet feature classifier head."""

import argparse

import torch.nn as nn

from tumor_classifier.config import DataPaths, TrainingConfig
from tumor_classifier.models.classifier import Classifier
from tumor_classifier.training.dataloaders import load_saved_features
from tumor_classifier.training.train import train_net
from tumor_classifier.utils.device import get_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    device = get_device()
    print(f"Using {device}")

    paths = DataPaths()
    config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
    )

    train_loader, val_loader, _ = load_saved_features(
        paths.feature_maps,
        batch_size=config.batch_size,
    )

    classifier = Classifier().to(device)
    train_net(classifier, train_loader, val_loader, device, config)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate a saved classifier checkpoint on the test feature set."""

import argparse

import torch
import torch.nn as nn

from tumor_classifier.config import DataPaths, TrainingConfig
from tumor_classifier.models.classifier import Classifier
from tumor_classifier.training.dataloaders import load_saved_features
from tumor_classifier.training.evaluate import evaluate, get_model_name
from tumor_classifier.utils.device import get_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--epoch", type=int, default=14)
    args = parser.parse_args()

    device = get_device()
    config = TrainingConfig(batch_size=args.batch_size, learning_rate=args.learning_rate)
    paths = DataPaths()

    _, _, test_loader = load_saved_features(
        paths.feature_maps,
        batch_size=config.batch_size,
    )

    net = Classifier()
    model_path = get_model_name(
        net.name,
        config.batch_size,
        config.learning_rate,
        args.epoch,
        checkpoint_dir=config.checkpoint_dir,
    )
    net.load_state_dict(torch.load(model_path, map_location=device))
    net = net.to(device)

    criterion = nn.BCEWithLogitsLoss()
    test_err, test_loss, f1_score, precision, recall = evaluate(
        net, test_loader, criterion, device
    )
    print(
        f"Test Accuracy: {1 - test_err}, F1-Score: {f1_score}, "
        f"Precision: {precision}, Recall: {recall}, Loss: {test_loss}"
    )


if __name__ == "__main__":
    main()

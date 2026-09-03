#!/usr/bin/env python3
"""Train and evaluate the linear SVM baseline model."""

import argparse

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from tumor_classifier.config import DataPaths
from tumor_classifier.models.svm_baseline import TumourClassifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    paths = DataPaths()
    transform = transforms.ToTensor()

    train_data = torchvision.datasets.ImageFolder(
        root=str(paths.augmented),
        transform=transform,
    )
    test_data = torchvision.datasets.ImageFolder(
        root=str(paths.dataset_split / "test"),
        transform=transform,
    )

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    classifier = TumourClassifier()
    classifier.train_svm(train_loader)
    print("Training complete.")

    metrics = classifier.evaluate(test_loader)
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test Precision: {metrics['precision']:.4f}")
    print(f"Test Recall: {metrics['recall']:.4f}")
    print(f"Test F1 Score: {metrics['f1']:.4f}")


if __name__ == "__main__":
    main()

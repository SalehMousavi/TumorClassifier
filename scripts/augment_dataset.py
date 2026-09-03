#!/usr/bin/env python3
"""Generate augmented training images for tumor and non-tumor classes."""

import argparse

import torchvision
import torchvision.transforms as transforms

from tumor_classifier.config import DataPaths
from tumor_classifier.data.augmentation import (
    augment_nontumor_images,
    augment_tumor_images,
    count_images_in_folder,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    paths = DataPaths()
    train_data = torchvision.datasets.ImageFolder(
        root=str(paths.dataset_split / "train"),
        transform=transforms.ToTensor(),
    )

    nontumor_count = augment_nontumor_images(
        train_data,
        paths.augmented / "NoTumor",
    )
    tumor_count = augment_tumor_images(
        train_data,
        paths.augmented / "Tumor",
    )

    print(f"Saved {nontumor_count} NoTumor images to {paths.augmented / 'NoTumor'}")
    print(f"Saved {tumor_count} Tumor images to {paths.augmented / 'Tumor'}")
    print(
        "Augmented folder counts:",
        count_images_in_folder(paths.augmented / "Tumor"),
        "tumor,",
        count_images_in_folder(paths.augmented / "NoTumor"),
        "non-tumor",
    )


if __name__ == "__main__":
    main()

"""Data augmentation for training set balancing."""

import random
from pathlib import Path

import torchvision
import torchvision.transforms as transforms


def get_augmentation_transforms():
    return [
        ("RandomRotation", transforms.RandomRotation(degrees=90)),
        ("RandomHorizontalFlip", transforms.RandomHorizontalFlip(p=1.0)),
        ("RandomVerticalFlip", transforms.RandomVerticalFlip(p=1.0)),
        ("RandomAffine", transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))),
        (
            "Color Jitter",
            transforms.ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5,
                hue=0.1,
            ),
        ),
        ("Gaussian Blur", transforms.GaussianBlur(kernel_size=5, sigma=(2, 5))),
    ]


def augment_nontumor_images(train_data, output_folder: Path) -> int:
    """Apply all augmentations to every NoTumor training image."""
    output_folder.mkdir(parents=True, exist_ok=True)
    transformations = get_augmentation_transforms()
    saved = 0

    for i, (img, label) in enumerate(train_data):
        if train_data.classes[label] != "NoTumor":
            continue

        original_image_path = output_folder / f"original_{i}.jpg"
        img.save(original_image_path)
        saved += 1

        for transform_name, transform in transformations:
            augmented_img = transform(img)
            augmented_image_path = output_folder / f"{transform_name}_{i}.jpg"
            augmented_img.save(augmented_image_path)
            saved += 1

    return saved


def augment_tumor_images(train_data, output_folder: Path) -> int:
    """Apply two random augmentations to every Tumor training image."""
    output_folder.mkdir(parents=True, exist_ok=True)
    transformations = get_augmentation_transforms()
    saved = 0

    for i, (img, label) in enumerate(train_data):
        if train_data.classes[label] != "Tumor":
            continue

        original_image_path = output_folder / f"original_{i}.jpg"
        img.save(original_image_path)
        saved += 1

        chosen_transforms = random.sample(transformations, 2)
        for j, (transform_name, transform) in enumerate(chosen_transforms, 1):
            augmented_img = transform(img)
            augmented_image_path = output_folder / f"{transform_name}_{i}_variant{j}.jpg"
            augmented_img.save(augmented_image_path)
            saved += 1

    return saved


def count_images_in_folder(folder_path: Path) -> int:
    return len(
        [
            f
            for f in folder_path.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        ]
    )

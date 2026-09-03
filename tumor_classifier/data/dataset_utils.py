"""Dataset merging and file copy helpers."""

import shutil
from pathlib import Path

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")


def copy_images(src_folder: Path, dst_folder: Path) -> int:
    """Copy image files from one folder to another."""
    dst_folder.mkdir(parents=True, exist_ok=True)
    copied = 0

    for filename in sorted(src_folder.iterdir()):
        if not filename.is_file():
            continue
        if filename.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        dst_path = dst_folder / filename.name
        shutil.copy(filename, dst_path)
        print(f"Copied: {filename} to {dst_path}")
        copied += 1

    return copied


def merge_datasets(paths) -> None:
    """Merge healthy, Kaggle tumor, and MATLAB tumor images into one dataset."""
    copy_images(paths.kaggle_healthy_224, paths.merged / "NoTumor")
    copy_images(paths.kaggle_tumor_224, paths.merged / "Tumor")
    copy_images(paths.matlab_processed_224, paths.merged / "Tumor")

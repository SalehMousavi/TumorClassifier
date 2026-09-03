"""Convert MATLAB .mat brain scan files to normalized JPEG images."""

from pathlib import Path

import cv2
import h5py
import numpy as np


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize int16 images to uint8 (0-255)."""
    image_min = np.min(image)
    image_max = np.max(image)
    image_normalized = (image - image_min) / (image_max - image_min)
    return (image_normalized * 255).astype(np.uint8)


def convert_matlab_directory(
    mat_files_dir: Path,
    output_images_dir: Path,
    *,
    jpeg_quality: int = 95,
) -> int:
    """Convert all valid 512x512 int16 .mat files in a directory to JPEG."""
    output_images_dir.mkdir(parents=True, exist_ok=True)
    files_added = 0

    for filename in sorted(mat_files_dir.iterdir()):
        if not filename.name.endswith(".mat"):
            continue

        with h5py.File(filename, "r") as h5_file:
            image_data = h5_file["cjdata"]["image"][:]

        if image_data.shape != (512, 512) or image_data.dtype != np.int16:
            print(f"Skipping file {filename.name}: Image is not 512x512 or not int16")
            continue

        image_normalized = normalize_image(image_data)
        output_image_path = output_images_dir / f"{filename.stem}.jpg"
        cv2.imwrite(
            str(output_image_path),
            image_normalized,
            [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
        )
        print(f"Saved {output_image_path}")
        files_added += 1

    print(f"Number of files added: {files_added}")
    return files_added

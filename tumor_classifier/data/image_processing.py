"""Resize and pad brain scan images to a standard size."""

from pathlib import Path

import cv2


def pad_image(image, target_size=(224, 224)):
    old_size = image.shape[:2]
    ratio = min(target_size[0] / old_size[0], target_size[1] / old_size[1])
    new_size = tuple(int(x * ratio) for x in old_size)

    resized_image = cv2.resize(image, (new_size[1], new_size[0]))

    delta_w = target_size[1] - new_size[1]
    delta_h = target_size[0] - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    return cv2.copyMakeBorder(
        resized_image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )


def process_image(image_path: Path, output_dir: Path, target_size=(224, 224)) -> bool:
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error loading image: {image_path}")
        return False

    output_filename = f"{image_path.stem}_processed.jpg"
    output_path = output_dir / output_filename
    processed_image = pad_image(image, target_size=target_size)
    cv2.imwrite(str(output_path), processed_image)
    print(f"Processed image saved as: {output_path}")
    return True


def process_directory(
    input_dir: Path,
    output_dir: Path,
    target_size=(224, 224),
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    files_added = 0

    for filename in sorted(input_dir.iterdir()):
        if not filename.is_file():
            continue
        if process_image(filename, output_dir, target_size=target_size):
            files_added += 1

    print(f"Number of files added: {files_added}")
    return files_added

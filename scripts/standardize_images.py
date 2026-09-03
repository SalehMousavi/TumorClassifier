#!/usr/bin/env python3
"""Resize and pad images in a directory to 224x224."""

import argparse
from pathlib import Path

from tumor_classifier.config import DataPaths
from tumor_classifier.data.image_processing import process_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    process_directory(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()

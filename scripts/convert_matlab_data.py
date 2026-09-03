#!/usr/bin/env python3
"""Convert MATLAB .mat brain scan files to JPEG images."""

import argparse
from pathlib import Path

from tumor_classifier.config import DataPaths
from tumor_classifier.data.matlab_conversion import convert_matlab_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=str,
        help="Directory containing .mat files (default: config matlab_raw path)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Directory for converted JPEG files (default: config matlab_processed path)",
    )
    args = parser.parse_args()

    paths = DataPaths()
    input_dir = paths.matlab_raw if args.input is None else Path(args.input)
    output_dir = paths.matlab_processed if args.output is None else Path(args.output)

    convert_matlab_directory(input_dir, output_dir)


if __name__ == "__main__":
    main()

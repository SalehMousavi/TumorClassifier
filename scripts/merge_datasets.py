#!/usr/bin/env python3
"""Merge healthy, Kaggle tumor, and MATLAB tumor datasets."""

from tumor_classifier.config import DataPaths
from tumor_classifier.data.dataset_utils import merge_datasets


def main():
    paths = DataPaths()
    merge_datasets(paths)
    print(f"Merged dataset written to {paths.merged}")


if __name__ == "__main__":
    main()

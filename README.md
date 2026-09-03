# TumorClassifier

Brain tumor classifier for APS360. Code is split into modules and scripts (the original notebook is in `legacy/`).

## Setup

```bash
pip install -r requirements.txt
pip install -e .
export TUMOR_DATA_ROOT=/path/to/data   # optional, defaults to ./data
```

## Run

```bash
python scripts/convert_matlab_data.py
python scripts/standardize_images.py --input ... --output ...
python scripts/merge_datasets.py
python scripts/augment_dataset.py
python scripts/extract_alexnet_features.py
python scripts/train_classifier.py
python scripts/evaluate_classifier.py --epoch 14
python scripts/train_svm_baseline.py
```

See `notebooks/pipeline.ipynb` for the notebook workflow.

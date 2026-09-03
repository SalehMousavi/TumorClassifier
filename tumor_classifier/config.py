"""Central configuration for dataset paths and training defaults."""

from dataclasses import dataclass, field
import os
from pathlib import Path


def _default_data_root() -> Path:
    return Path(os.environ.get("TUMOR_DATA_ROOT", "data"))


@dataclass
class DataPaths:
    """Filesystem layout for the tumor classification pipeline."""

    root: Path = field(default_factory=_default_data_root)

    @property
    def matlab_raw(self) -> Path:
        return self.root / "brainTumorDataUnprocessedMatlab4"

    @property
    def matlab_processed(self) -> Path:
        return self.root / "ProccessedMatlabData"

    @property
    def kaggle_healthy_raw(self) -> Path:
        return self.root / "KaggleDataSet" / "Brain Tumor Data Set" / "Healthy"

    @property
    def kaggle_healthy_224(self) -> Path:
        return self.root / "KaggleHealthy224x224"

    @property
    def kaggle_tumor_224(self) -> Path:
        return self.root / "KaggleBrainTumor224x224"

    @property
    def matlab_processed_224(self) -> Path:
        return self.root / "ProccessedMatlabData224x224"

    @property
    def merged(self) -> Path:
        return self.root / "MergedDataSet224x224"

    @property
    def dataset_split(self) -> Path:
        return self.root / "DatasetSplit"

    @property
    def augmented(self) -> Path:
        return self.root / "Augmented"

    @property
    def feature_maps(self) -> Path:
        return self.root / "AlexNetFeatureMaps"


@dataclass
class TrainingConfig:
    """Default hyperparameters used across training scripts."""

    batch_size: int = 32
    learning_rate: float = 0.01
    num_epochs: int = 30
    image_size: tuple[int, int] = (224, 224)
    random_seed: int = 1000
    checkpoint_dir: Path = Path("checkpoints")

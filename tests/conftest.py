import os

import hydra
import pytest
from omegaconf import DictConfig

from proj.data import preprocess_subset
from proj.train import train


@pytest.fixture(scope="session")
def cfg() -> DictConfig:
    os.environ["WANDB_MODE"] = "offline"

    hydra.initialize(version_base=None, config_path="../configs/hydra")
    cfg = hydra.compose(
        config_name="config.yaml",
        overrides=[
            "data_dir=data/processed_test",
            "model.num_classes=5",
            "print_progress=False",
        ],
    )

    # preprocess and train on a very small amount of data to have valid processed data and model to run tests on
    preprocess_subset(raw_dir="data/raw_test", processed_dir="data/processed_test")
    train(cfg)

    return cfg

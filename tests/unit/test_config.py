from pathlib import Path

import pytest
from pydantic import ValidationError

from forgelens.config import ExperimentConfig


def valid_payload() -> dict[str, object]:
    return {
        "experiment_id": "BASE-RGB-001",
        "hypothesis": "A tiny RGB model learns a non-trivial baseline.",
        "seed": 7,
        "device": "auto",
        "model_name": "tiny_joint",
        "base_channels": 8,
        "data": {
            "dataset_name": "fixture",
            "dataset_version": "v1",
            "root": Path("data/fixture"),
            "split_manifest": Path("data/manifest.json"),
            "image_size": (64, 96),
        },
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 0.001,
        },
    }


def test_experiment_config_validates() -> None:
    config = ExperimentConfig.model_validate(valid_payload())
    assert config.training.batch_size == 2


def test_experiment_config_rejects_tiny_images() -> None:
    payload = valid_payload()
    data = payload["data"]
    assert isinstance(data, dict)
    data["image_size"] = (16, 16)
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(payload)

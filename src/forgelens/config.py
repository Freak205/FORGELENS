"""Validated experiment configuration."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class TrainingConfig(BaseModel):
    """Training controls that affect reproducibility and memory."""

    epochs: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    learning_rate: float = Field(gt=0.0)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    mixed_precision: bool = True
    checkpoint_every_epochs: int = Field(default=1, ge=1)


class DataConfig(BaseModel):
    """Data paths and split rules."""

    dataset_name: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    root: Path
    split_manifest: Path
    image_size: tuple[int, int]
    group_key: str = "source_image_id"

    @model_validator(mode="after")
    def validate_image_size(self) -> "DataConfig":
        if any(dimension < 32 for dimension in self.image_size):
            raise ValueError("each image dimension must be at least 32")
        return self


class ExperimentConfig(BaseModel):
    """Complete immutable input to a train/evaluate run."""

    experiment_id: str = Field(pattern=r"^[A-Z][A-Z0-9-]+$")
    hypothesis: str = Field(min_length=10)
    seed: int = Field(ge=0)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    model_name: Literal["tiny_joint"] = "tiny_joint"
    base_channels: int = Field(default=16, ge=2)
    data: DataConfig
    training: TrainingConfig

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        """Load and validate a YAML configuration."""
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("experiment config must be a YAML mapping")
        return cls.model_validate(payload)

"""Traceable and resumable checkpoint utilities."""

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    metrics: dict[str, float],
    config: dict[str, Any],
) -> None:
    """Atomically save a training checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format_version": 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "config": config,
        },
        temporary_path,
    )
    temporary_path.replace(path)


def load_checkpoint(
    path: Path, model: nn.Module, optimizer: Optimizer
) -> dict[str, Any]:
    """Restore model and optimizer from a local trusted checkpoint."""
    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    return payload

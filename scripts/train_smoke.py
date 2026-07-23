"""Train the tiny joint baseline on safe fictional fixtures."""

import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from forgelens.data import DocumentSample, FictionalDocumentFixtures
from forgelens.evaluation import binary_metrics, pixel_iou
from forgelens.models import TinyJointDetector
from forgelens.training import save_checkpoint, seed_everything

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "SMOKE-0001"
SEED = 20260723


def collate(samples: list[DocumentSample]) -> tuple[Tensor, Tensor, Tensor]:
    """Stack fixture dataclasses into tensors."""
    return (
        torch.stack([sample.image for sample in samples]),
        torch.stack([sample.mask for sample in samples]),
        torch.stack([sample.label for sample in samples]),
    )


def git_commit() -> str:
    """Return current commit or an explicit uncommitted marker."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "UNCOMMITTED"


def main() -> None:
    """Run one deterministic smoke epoch and persist traceable artifacts."""
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = FictionalDocumentFixtures(size=12)
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
        collate_fn=collate,
    )
    model = TinyJointDetector(base_channels=8).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    classification_loss = nn.BCEWithLogitsLoss()
    localization_loss = nn.BCEWithLogitsLoss()
    started = time.perf_counter()
    model.train()
    total_loss = 0.0
    for images, masks, labels in loader:
        images, masks, labels = images.to(device), masks.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(images)
        loss = classification_loss(output.image_logits, labels)
        loss = loss + localization_loss(output.mask_logits, masks)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite smoke-training loss")
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())

    duration = time.perf_counter() - started
    model.eval()
    with torch.no_grad():
        images, masks, labels = collate(
            [dataset[index] for index in range(len(dataset))]
        )
        output = model(images.to(device))
        probabilities = output.image_logits.sigmoid().cpu()
        mask_probabilities = output.mask_logits.sigmoid().cpu()
    metrics = {
        "mean_train_loss": total_loss / len(loader),
        "image_f1_at_0_5": binary_metrics(probabilities, labels).f1,
        "pixel_iou_at_0_5": pixel_iou(mask_probabilities, masks),
        "duration_seconds": duration,
        "peak_vram_mb": (
            torch.cuda.max_memory_allocated() / 1048576
            if device.type == "cuda"
            else 0.0
        ),
    }
    config: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "purpose": "pipeline smoke only; not a research result",
        "seed": SEED,
        "epochs": 1,
        "batch_size": 4,
        "dataset": "FictionalDocumentFixtures(size=12)",
        "model": "TinyJointDetector(base_channels=8)",
        "optimizer": "AdamW(lr=1e-3)",
    }
    output_directory = ROOT / "artifacts" / "experiments" / EXPERIMENT_ID
    output_directory.mkdir(parents=True, exist_ok=True)
    save_checkpoint(output_directory / "last.pt", model, optimizer, 1, metrics, config)
    record = {
        "experiment_id": EXPERIMENT_ID,
        "timestamp_unix": time.time(),
        "hypothesis": "joint baseline can train end-to-end on safe fixtures",
        "dataset_version": "blank-canvas-fixtures-v1",
        "split_manifest": "not applicable: pipeline smoke only",
        "config": config,
        "seed": SEED,
        "git_commit": git_commit(),
        "hardware": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
        "metrics": metrics,
        "checkpoint": str(output_directory / "last.pt"),
        "decision": "verify training/checkpoint path; do not publish as performance",
    }
    (output_directory / "record.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

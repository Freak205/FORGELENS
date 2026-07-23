"""Train and evaluate the first real-document RGB baseline."""

from __future__ import annotations

import json
import argparse
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.amp import GradScaler, autocast  # type: ignore[attr-defined]
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from forgelens.calibration import (
    TemperatureScaler,
    brier_score,
    expected_calibration_error,
    validation_optimal_threshold,
)
from forgelens.config import ExperimentConfig
from forgelens.data import DocumentSample, ManifestDocumentDataset
from forgelens.evaluation import (
    binary_metrics,
    bootstrap_interval,
    pixel_iou,
    pr_auc,
    roc_auc,
)
from forgelens.models import TinyJointDetector, TinyUNetJointDetector
from forgelens.training import load_checkpoint, save_checkpoint, seed_everything

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "training" / "cord_copy_move_rgb.yaml"
JointModel = TinyJointDetector | TinyUNetJointDetector


class CachedDocumentDataset(Dataset[DocumentSample]):
    """In-memory resized samples to avoid repeated high-resolution PNG decode."""

    def __init__(self, samples: list[DocumentSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> DocumentSample:
        return self.samples[index]


def collate(samples: list[DocumentSample]) -> tuple[Tensor, Tensor, Tensor]:
    return (
        torch.stack([sample.image for sample in samples]),
        torch.stack([sample.mask for sample in samples]),
        torch.stack([sample.label for sample in samples]),
    )


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "UNCOMMITTED"


def localization_loss(logits: Tensor, targets: Tensor) -> Tensor:
    positive_weight = torch.tensor([20.0], device=logits.device)
    binary = nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=positive_weight,
    )
    probabilities = logits.sigmoid()
    axes = tuple(range(1, targets.ndim))
    intersection = (probabilities * targets).sum(dim=axes)
    denominator = probabilities.sum(dim=axes) + targets.sum(dim=axes)
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return 0.5 * binary + 0.5 * dice


def make_loader(
    dataset: Dataset[DocumentSample],
    batch_size: int,
    seed: int,
    *,
    shuffle: bool,
) -> DataLoader[DocumentSample]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(seed),
        collate_fn=collate,
    )


def train_epoch(
    model: JointModel,
    loader: DataLoader[DocumentSample],
    optimizer: AdamW,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.train()
    total = 0.0
    for images, masks, labels in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=use_amp):
            output = model(images)
            classification = nn.functional.binary_cross_entropy_with_logits(
                output.image_logits,
                labels,
            )
            loss = classification + localization_loss(output.mask_logits, masks)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite training loss")
        scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.item())
    return total / len(loader)


@torch.no_grad()
def predict(
    model: JointModel,
    loader: DataLoader[DocumentSample],
    device: torch.device,
    use_amp: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    model.eval()
    logits: list[Tensor] = []
    labels: list[Tensor] = []
    mask_probabilities: list[Tensor] = []
    masks: list[Tensor] = []
    for images, batch_masks, batch_labels in loader:
        with autocast(device_type=device.type, enabled=use_amp):
            output = model(images.to(device, non_blocking=True))
        logits.append(output.image_logits.float().cpu())
        labels.append(batch_labels)
        mask_probabilities.append(output.mask_logits.sigmoid().float().cpu())
        masks.append(batch_masks)
    return (
        torch.cat(logits),
        torch.cat(labels),
        torch.cat(mask_probabilities),
        torch.cat(masks),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    arguments = parser.parse_args()
    config_path = (
        arguments.config
        if arguments.config.is_absolute()
        else PROJECT_ROOT / arguments.config
    )
    config = ExperimentConfig.from_yaml(config_path)
    seed_everything(config.seed)
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda"
        if config.device == "cuda"
        or (config.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    use_amp = config.training.mixed_precision and device.type == "cuda"
    manifest_path = PROJECT_ROOT / config.data.split_manifest
    source_datasets = {
        split: ManifestDocumentDataset(
            manifest_path,
            config.data.root,
            split,
            config.data.image_size,
        )
        for split in ("train", "validation", "test")
    }
    datasets = {
        split: CachedDocumentDataset([dataset[index] for index in range(len(dataset))])
        for split, dataset in source_datasets.items()
    }
    loaders = {
        split: make_loader(
            dataset,
            config.training.batch_size,
            config.seed,
            shuffle=split == "train",
        )
        for split, dataset in datasets.items()
    }
    model: JointModel
    if config.model_name == "tiny_joint":
        model = TinyJointDetector(config.base_channels)
    else:
        model = TinyUNetJointDetector(config.base_channels)
    model = model.to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=1e-4,
    )
    scaler = GradScaler(device.type, enabled=use_amp)
    output_directory = PROJECT_ROOT / "artifacts" / "experiments" / config.experiment_id
    resolved_config: dict[str, Any] = config.model_dump(mode="json")
    started = time.perf_counter()
    history: list[dict[str, float | int]] = []
    best_validation_auc = -1.0
    for epoch in range(1, config.training.epochs + 1):
        train_loss = train_epoch(
            model,
            loaders["train"],
            optimizer,
            scaler,
            device,
            use_amp,
        )
        validation_logits, validation_labels, _, _ = predict(
            model,
            loaders["validation"],
            device,
            use_amp,
        )
        validation_auc = roc_auc(validation_logits.sigmoid(), validation_labels)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_roc_auc": validation_auc,
            }
        )
        print(json.dumps(history[-1]))
        if validation_auc >= best_validation_auc:
            best_validation_auc = validation_auc
            save_checkpoint(
                output_directory / "best.pt",
                model,
                optimizer,
                epoch,
                {
                    "train_loss": train_loss,
                    "validation_roc_auc": validation_auc,
                },
                resolved_config,
            )

    load_checkpoint(output_directory / "best.pt", model, optimizer)
    validation_logits, validation_labels, _, _ = predict(
        model,
        loaders["validation"],
        device,
        use_amp,
    )
    temperature = TemperatureScaler()
    temperature.fit(validation_logits, validation_labels)
    validation_probabilities = temperature(validation_logits).sigmoid()
    threshold, validation_f1 = validation_optimal_threshold(
        validation_probabilities,
        validation_labels,
    )
    test_logits, test_labels, test_mask_probabilities, test_masks = predict(
        model,
        loaders["test"],
        device,
        use_amp,
    )
    test_probabilities = temperature(test_logits).sigmoid()
    classification = binary_metrics(test_probabilities, test_labels, threshold)
    forged = test_labels.bool()
    roc_interval = bootstrap_interval(
        roc_auc,
        test_probabilities,
        test_labels,
        samples=500,
        seed=config.seed,
    )
    pr_interval = bootstrap_interval(
        pr_auc,
        test_probabilities,
        test_labels,
        samples=500,
        seed=config.seed + 1,
    )
    metrics: dict[str, Any] = {
        "validation_selected_threshold": threshold,
        "validation_f1": validation_f1,
        "temperature": float(temperature.temperature.item()),
        "test": {
            "roc_auc": asdict(roc_interval),
            "pr_auc": asdict(pr_interval),
            "classification": asdict(classification),
            "ece_15_bin": expected_calibration_error(
                test_probabilities,
                test_labels,
            ),
            "brier": brier_score(test_probabilities, test_labels),
            "forged_pixel_iou_at_0_5": pixel_iou(
                test_mask_probabilities[forged],
                test_masks[forged],
            ),
        },
        "duration_seconds": time.perf_counter() - started,
        "peak_vram_mb": (
            torch.cuda.max_memory_allocated() / 1048576
            if device.type == "cuda"
            else 0.0
        ),
    }
    record = {
        "experiment_id": config.experiment_id,
        "timestamp_unix": time.time(),
        "hypothesis": config.hypothesis,
        "dataset_version": config.data.dataset_version,
        "split_manifest": str(manifest_path),
        "config": resolved_config,
        "seed": config.seed,
        "git_commit": git_commit(),
        "hardware": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
        "history": history,
        "metrics": metrics,
        "checkpoint": str(output_directory / "best.pt"),
        "limitations": (
            "Traditional deterministic copy-move benchmark only; not evidence "
            "of AI-inpainting or in-the-wild performance."
        ),
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "record.json").write_text(
        json.dumps(record, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

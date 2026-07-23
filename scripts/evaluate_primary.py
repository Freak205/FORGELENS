"""Evaluate few-shot calibration, abstention, and capture robustness."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from forgelens.calibration import (
    TemperatureScaler,
    brier_score,
    expected_calibration_error,
    validation_optimal_threshold,
)
from forgelens.config import ExperimentConfig
from forgelens.data import DocumentSample, ManifestDocumentDataset
from forgelens.evaluation import binary_metrics, roc_auc
from forgelens.models import TinyUNetJointDetector
from forgelens.robustness import CorruptionName, apply_corruption

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "training" / "aiforge_v2_cord_unet.yaml"
EXPERIMENT = PROJECT_ROOT / "artifacts" / "experiments" / "AIFORGE-CORD-UNET-001"
SHOT_COUNTS = (1, 5, 10, 25)
CORRUPTIONS: tuple[CorruptionName, ...] = (
    "jpeg",
    "blur",
    "rotation",
    "perspective",
    "low_illumination",
    "noise",
    "resize",
    "screenshot",
    "print_scan",
)


def collate(samples: list[DocumentSample]) -> tuple[Tensor, Tensor]:
    return (
        torch.stack([sample.image for sample in samples]),
        torch.stack([sample.label for sample in samples]),
    )


class CachedDataset(Dataset[DocumentSample]):
    """Typed wrapper around decoded evaluation samples."""

    def __init__(self, samples: list[DocumentSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> DocumentSample:
        return self.samples[index]


@torch.no_grad()
def predict(
    model: TinyUNetJointDetector,
    samples: list[DocumentSample],
    device: torch.device,
    corruption: CorruptionName | None = None,
) -> tuple[Tensor, Tensor]:
    loader: DataLoader[DocumentSample] = DataLoader(
        CachedDataset(samples),
        batch_size=8,
        collate_fn=collate,
    )
    logits: list[Tensor] = []
    labels: list[Tensor] = []
    model.eval()
    for batch_images, batch_labels in loader:
        if corruption is not None:
            batch_images = torch.stack(
                [
                    apply_corruption(
                        image,
                        corruption,
                        severity=3,
                        seed=20260723 + index,
                    )
                    for index, image in enumerate(batch_images)
                ]
            )
        output = model(batch_images.to(device))
        logits.append(output.image_logits.float().cpu())
        labels.append(batch_labels)
    return torch.cat(logits), torch.cat(labels)


def risk_coverage(
    probabilities: Tensor,
    labels: Tensor,
    threshold: float,
) -> list[dict[str, float]]:
    """Return selective error at fixed coverage levels."""
    confidence = (probabilities - threshold).abs()
    order = confidence.argsort(descending=True)
    predictions = probabilities >= threshold
    results = []
    for coverage in (0.5, 0.75, 0.9, 1.0):
        count = max(1, round(len(labels) * coverage))
        selected = order[:count]
        error = float((predictions[selected] != labels[selected].bool()).float().mean())
        results.append(
            {
                "coverage": coverage,
                "selective_error": error,
                "retained_samples": float(count),
            }
        )
    return results


def calibration_record(
    validation_logits: Tensor,
    validation_labels: Tensor,
    test_logits: Tensor,
    test_labels: Tensor,
) -> tuple[dict[str, Any], TemperatureScaler, float]:
    scaler = TemperatureScaler()
    scaler.fit(validation_logits, validation_labels)
    probabilities = scaler(test_logits).sigmoid()
    validation_probabilities = scaler(validation_logits).sigmoid()
    threshold, _ = validation_optimal_threshold(
        validation_probabilities,
        validation_labels,
    )
    classification = binary_metrics(probabilities, test_labels, threshold)
    return (
        {
            "temperature": float(scaler.temperature.item()),
            "threshold": threshold,
            "roc_auc": roc_auc(probabilities, test_labels),
            "classification": asdict(classification),
            "ece_15_bin": expected_calibration_error(probabilities, test_labels),
            "brier": brier_score(probabilities, test_labels),
            "risk_coverage": risk_coverage(probabilities, test_labels, threshold),
        },
        scaler,
        threshold,
    )


def main() -> None:
    config = ExperimentConfig.from_yaml(CONFIG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyUNetJointDetector(config.base_channels).to(device)
    checkpoint = torch.load(
        EXPERIMENT / "best.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model"])
    manifest_path = PROJECT_ROOT / config.data.split_manifest
    validation_dataset = ManifestDocumentDataset(
        manifest_path,
        config.data.root,
        "validation",
        config.data.image_size,
    )
    test_dataset = ManifestDocumentDataset(
        manifest_path,
        config.data.root,
        "test",
        config.data.image_size,
    )
    validation_samples = [
        validation_dataset[index] for index in range(len(validation_dataset))
    ]
    test_samples = [test_dataset[index] for index in range(len(test_dataset))]
    validation_logits, validation_labels = predict(model, validation_samples, device)
    test_logits, test_labels = predict(model, test_samples, device)

    by_group: dict[str, list[int]] = {}
    for index, sample in enumerate(validation_samples):
        by_group.setdefault(sample.source_group, []).append(index)
    groups = sorted(by_group)
    few_shot: dict[str, Any] = {}
    for shots in SHOT_COUNTS:
        indices = [index for group in groups[:shots] for index in by_group[group]]
        record, _, _ = calibration_record(
            validation_logits[indices],
            validation_labels[indices],
            test_logits,
            test_labels,
        )
        few_shot[str(shots)] = record

    full_record, scaler, threshold = calibration_record(
        validation_logits,
        validation_labels,
        test_logits,
        test_labels,
    )
    clean_auc = float(full_record["roc_auc"])
    robustness: dict[str, Any] = {}
    for corruption in CORRUPTIONS:
        corrupted_logits, labels = predict(
            model,
            test_samples,
            device,
            corruption,
        )
        probabilities = scaler(corrupted_logits).sigmoid()
        auc = roc_auc(probabilities, labels)
        robustness[corruption] = {
            "severity": 3,
            "roc_auc": auc,
            "delta_from_clean": auc - clean_auc,
            "classification": asdict(binary_metrics(probabilities, labels, threshold)),
        }
    output = {
        "experiment_id": "AIFORGE-CORD-UNET-001-EVAL",
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "few_shot_group_calibration": few_shot,
        "full_validation_calibration": full_record,
        "capture_robustness": robustness,
        "limitations": (
            "Few-shot groups are deterministic CORD validation pairs. "
            "Corruptions are synthetic severity-3 capture proxies."
        ),
    }
    (EXPERIMENT / "evaluation.json").write_text(
        json.dumps(output, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

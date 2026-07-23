import pytest
import torch

from forgelens.calibration import (
    brier_score,
    expected_calibration_error,
    validation_optimal_pixel_threshold,
    validation_optimal_threshold,
)
from forgelens.evaluation import binary_metrics, pixel_iou


def test_binary_metrics_perfect_predictions() -> None:
    metrics = binary_metrics(
        torch.tensor([0.1, 0.9]), torch.tensor([0, 1]), threshold=0.5
    )
    assert metrics.f1 == pytest.approx(1.0)
    assert metrics.false_positive_rate == 0.0
    assert metrics.false_negative_rate == 0.0


def test_pixel_iou_empty_masks_is_one() -> None:
    assert pixel_iou(torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 2)) == 1.0


def test_calibration_metrics_and_threshold() -> None:
    probabilities = torch.tensor([0.1, 0.4, 0.8, 0.9])
    targets = torch.tensor([0, 0, 1, 1])
    assert brier_score(probabilities, targets) == pytest.approx(0.055)
    assert expected_calibration_error(probabilities, targets, bins=2) >= 0.0
    threshold, f1 = validation_optimal_threshold(probabilities, targets, steps=11)
    assert threshold == pytest.approx(0.8)
    assert f1 == pytest.approx(1.0)


def test_validation_optimal_pixel_threshold() -> None:
    probabilities = torch.tensor([[[[0.1, 0.3], [0.7, 0.9]]]])
    targets = torch.tensor([[[[0, 0], [1, 1]]]])
    threshold, iou = validation_optimal_pixel_threshold(
        probabilities,
        targets,
        steps=11,
    )
    assert threshold == pytest.approx(0.7)
    assert iou == pytest.approx(1.0)

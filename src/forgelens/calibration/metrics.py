"""Calibration metrics and validation-only threshold selection."""

import torch
from torch import Tensor

from forgelens.evaluation import binary_metrics


def expected_calibration_error(
    probabilities: Tensor, targets: Tensor, bins: int = 15
) -> float:
    """Return equal-width expected calibration error."""
    if bins < 1:
        raise ValueError("bins must be positive")
    boundaries = torch.linspace(0.0, 1.0, bins + 1, device=probabilities.device)
    error = torch.zeros((), device=probabilities.device)
    truth = targets.float()
    for index in range(bins):
        lower = boundaries[index]
        upper = boundaries[index + 1]
        in_bin = (probabilities > lower) & (probabilities <= upper)
        if index == 0:
            in_bin = (probabilities >= lower) & (probabilities <= upper)
        if in_bin.any():
            confidence = probabilities[in_bin].mean()
            accuracy = truth[in_bin].mean()
            error += in_bin.float().mean() * (confidence - accuracy).abs()
    return float(error.item())


def brier_score(probabilities: Tensor, targets: Tensor) -> float:
    """Return mean squared probability error."""
    return float(torch.mean((probabilities - targets.float()) ** 2).item())


def validation_optimal_threshold(
    probabilities: Tensor, targets: Tensor, steps: int = 101
) -> tuple[float, float]:
    """Select the F1-optimal threshold on validation data."""
    if steps < 2:
        raise ValueError("steps must be at least two")
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold_tensor in torch.linspace(0.0, 1.0, steps):
        threshold = float(threshold_tensor.item())
        f1 = binary_metrics(probabilities, targets, threshold).f1
        if f1 >= best_f1:
            best_threshold = threshold
            best_f1 = f1
    return best_threshold, best_f1

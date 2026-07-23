"""Dependency-light binary classification and localization metrics."""

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class BinaryMetrics:
    """Thresholded binary metrics."""

    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    false_negative_rate: float


def _safe_ratio(numerator: Tensor, denominator: Tensor) -> float:
    return float((numerator / denominator.clamp_min(1)).item())


def binary_metrics(
    probabilities: Tensor, targets: Tensor, threshold: float = 0.5
) -> BinaryMetrics:
    """Compute stable binary metrics from one-dimensional tensors."""
    predictions = probabilities >= threshold
    truth = targets.bool()
    true_positive = (predictions & truth).sum()
    false_positive = (predictions & ~truth).sum()
    false_negative = (~predictions & truth).sum()
    true_negative = (~predictions & ~truth).sum()
    precision = _safe_ratio(true_positive, true_positive + false_positive)
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    denominator = precision + recall
    f1 = 0.0 if denominator == 0.0 else 2.0 * precision * recall / denominator
    return BinaryMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        false_positive_rate=_safe_ratio(false_positive, false_positive + true_negative),
        false_negative_rate=_safe_ratio(false_negative, false_negative + true_positive),
    )


def pixel_iou(probabilities: Tensor, targets: Tensor, threshold: float = 0.5) -> float:
    """Compute foreground intersection-over-union over a batch."""
    predictions = probabilities >= threshold
    truth = targets.bool()
    intersection = (predictions & truth).sum()
    union = (predictions | truth).sum()
    if int(union.item()) == 0:
        return 1.0
    return float((intersection / union).item())

"""Ranking metrics and deterministic bootstrap confidence intervals."""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor


def roc_auc(probabilities: Tensor, targets: Tensor) -> float:
    """Compute binary ROC-AUC using pairwise comparison with tie handling."""
    positive = probabilities[targets.bool()]
    negative = probabilities[~targets.bool()]
    if positive.numel() == 0 or negative.numel() == 0:
        raise ValueError("ROC-AUC requires both classes")
    comparisons = positive[:, None] - negative[None, :]
    auc = (comparisons > 0).float().mean()
    auc += 0.5 * (comparisons == 0).float().mean()
    return float(auc.item())


def pr_auc(probabilities: Tensor, targets: Tensor) -> float:
    """Compute average precision, a stepwise PR-AUC."""
    positive_count = int(targets.bool().sum().item())
    if positive_count == 0:
        raise ValueError("PR-AUC requires positive samples")
    order = torch.argsort(probabilities, descending=True, stable=True)
    sorted_targets = targets[order].float()
    true_positives = torch.cumsum(sorted_targets, dim=0)
    precision = true_positives / torch.arange(
        1, len(targets) + 1, device=targets.device
    )
    return float((precision * sorted_targets).sum().item() / positive_count)


@dataclass(frozen=True)
class ConfidenceInterval:
    """Point estimate and percentile bootstrap interval."""

    estimate: float
    lower: float
    upper: float
    samples: int


def bootstrap_interval(
    metric: Callable[[Tensor, Tensor], float],
    probabilities: Tensor,
    targets: Tensor,
    samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 20260723,
) -> ConfidenceInterval:
    """Return deterministic percentile bootstrap confidence bounds."""
    if samples < 20:
        raise ValueError("at least 20 bootstrap samples are required")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    generator = torch.Generator(device=probabilities.device).manual_seed(seed)
    collected: list[float] = []
    for _ in range(samples):
        indices = torch.randint(
            len(targets),
            (len(targets),),
            generator=generator,
            device=targets.device,
        )
        try:
            collected.append(metric(probabilities[indices], targets[indices]))
        except ValueError:
            continue
    if len(collected) < samples // 2:
        raise ValueError("too many invalid bootstrap resamples")
    values = torch.tensor(collected)
    alpha = (1.0 - confidence) / 2.0
    return ConfidenceInterval(
        estimate=metric(probabilities, targets),
        lower=float(torch.quantile(values, alpha).item()),
        upper=float(torch.quantile(values, 1.0 - alpha).item()),
        samples=len(collected),
    )

"""Uncertainty-aware decision policy."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class OperatingPolicy:
    """Two-threshold policy with an explicit review region."""

    accept_below: float
    reject_at_or_above: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.accept_below < self.reject_at_or_above <= 1.0:
            raise ValueError("policy thresholds must define a valid review interval")

    def decide(
        self, calibrated_risk: float
    ) -> tuple[
        Literal["authentic", "forged", "uncertain"],
        Literal["accept", "reject", "manual_review"],
    ]:
        """Map calibrated risk to verdict and recommended action."""
        if not 0.0 <= calibrated_risk <= 1.0:
            raise ValueError("calibrated risk must be between zero and one")
        if calibrated_risk < self.accept_below:
            return "authentic", "accept"
        if calibrated_risk >= self.reject_at_or_above:
            return "forged", "reject"
        return "uncertain", "manual_review"

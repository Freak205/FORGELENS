"""Trainable ForgeLens models."""

from forgelens.models.baselines import (
    DetectorOutput,
    ResidualUNetJointDetector,
    TinyJointDetector,
    TinyUNetJointDetector,
)

__all__ = [
    "DetectorOutput",
    "ResidualUNetJointDetector",
    "TinyJointDetector",
    "TinyUNetJointDetector",
]

"""Leakage-safe grouped split helpers."""

from collections.abc import Iterable
from dataclasses import dataclass
from random import Random


@dataclass(frozen=True)
class GroupedSplit:
    """Group identifiers assigned to mutually exclusive partitions."""

    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]


def grouped_split(
    groups: Iterable[str],
    seed: int,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> GroupedSplit:
    """Split unique source groups without cross-partition leakage."""
    if not 0.0 < validation_fraction < 1.0 or not 0.0 < test_fraction < 1.0:
        raise ValueError("split fractions must be between zero and one")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation and test fractions must sum below one")
    unique_groups = sorted(set(groups))
    if len(unique_groups) < 3:
        raise ValueError("at least three source groups are required")
    Random(seed).shuffle(unique_groups)
    validation_count = max(1, round(len(unique_groups) * validation_fraction))
    test_count = max(1, round(len(unique_groups) * test_fraction))
    if validation_count + test_count >= len(unique_groups):
        raise ValueError("too few groups for the requested fractions")
    return GroupedSplit(
        train=tuple(unique_groups[validation_count + test_count :]),
        validation=tuple(unique_groups[:validation_count]),
        test=tuple(unique_groups[validation_count : validation_count + test_count]),
    )

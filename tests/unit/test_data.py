import pytest
import torch

from forgelens.data import FictionalDocumentFixtures, grouped_split


def test_fixtures_are_deterministic_and_masked() -> None:
    first = FictionalDocumentFixtures(size=4)
    second = FictionalDocumentFixtures(size=4)
    assert torch.equal(first[1].image, second[1].image)
    assert first[0].mask.sum() == 0
    assert first[1].mask.sum() > 0
    assert first[0].label.item() == 0.0
    assert first[1].label.item() == 1.0


def test_grouped_split_has_no_leakage() -> None:
    split = grouped_split([f"group-{index}" for index in range(10)], seed=7)
    assert not (set(split.train) & set(split.validation))
    assert not (set(split.train) & set(split.test))
    assert not (set(split.validation) & set(split.test))
    assert len(split.train) + len(split.validation) + len(split.test) == 10


def test_grouped_split_rejects_too_few_groups() -> None:
    with pytest.raises(ValueError):
        grouped_split(["same", "same"], seed=1)

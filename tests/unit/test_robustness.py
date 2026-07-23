import pytest
import torch

from forgelens.robustness import CorruptionName, apply_corruption


@pytest.mark.parametrize(
    "name",
    [
        "jpeg",
        "blur",
        "rotation",
        "perspective",
        "low_illumination",
        "noise",
        "resize",
        "screenshot",
        "print_scan",
    ],
)
def test_corruptions_preserve_shape_and_range(name: CorruptionName) -> None:
    image = torch.rand(3, 48, 64)
    corrupted = apply_corruption(image, name, severity=3, seed=7)
    assert corrupted.shape == image.shape
    assert float(corrupted.min()) >= 0.0
    assert float(corrupted.max()) <= 1.0


def test_noise_is_deterministic() -> None:
    image = torch.rand(3, 48, 64)
    first = apply_corruption(image, "noise", severity=2, seed=7)
    second = apply_corruption(image, "noise", severity=2, seed=7)
    assert torch.equal(first, second)


def test_invalid_severity_is_rejected() -> None:
    with pytest.raises(ValueError):
        apply_corruption(torch.rand(3, 48, 64), "blur", severity=0)

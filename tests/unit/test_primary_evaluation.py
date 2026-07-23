import torch

from scripts.evaluate_primary import risk_coverage


def test_risk_coverage_retains_more_samples_monotonically() -> None:
    probabilities = torch.tensor([0.01, 0.2, 0.55, 0.8, 0.99])
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0])
    curve = risk_coverage(probabilities, labels, threshold=0.5)

    assert [point["coverage"] for point in curve] == [0.5, 0.75, 0.9, 1.0]
    assert [point["retained_samples"] for point in curve] == [2.0, 4.0, 4.0, 5.0]

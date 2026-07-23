import pytest
import torch

from forgelens.calibration import OperatingPolicy
from forgelens.evaluation import bootstrap_interval, pr_auc, roc_auc


def test_ranking_metrics_are_perfect() -> None:
    probabilities = torch.tensor([0.1, 0.2, 0.8, 0.9])
    targets = torch.tensor([0, 0, 1, 1])
    assert roc_auc(probabilities, targets) == 1.0
    assert pr_auc(probabilities, targets) == 1.0


def test_roc_auc_ties_are_half_credit() -> None:
    assert roc_auc(torch.tensor([0.5, 0.5]), torch.tensor([0, 1])) == 0.5


def test_bootstrap_interval_is_bounded() -> None:
    probabilities = torch.tensor([0.1, 0.2, 0.8, 0.9])
    targets = torch.tensor([0, 0, 1, 1])
    interval = bootstrap_interval(roc_auc, probabilities, targets, samples=50, seed=1)
    assert 0.0 <= interval.lower <= interval.estimate <= interval.upper <= 1.0


def test_operating_policy_abstains_between_thresholds() -> None:
    policy = OperatingPolicy(accept_below=0.2, reject_at_or_above=0.8)
    assert policy.decide(0.1) == ("authentic", "accept")
    assert policy.decide(0.5) == ("uncertain", "manual_review")
    assert policy.decide(0.8) == ("forged", "reject")
    with pytest.raises(ValueError):
        policy.decide(1.1)

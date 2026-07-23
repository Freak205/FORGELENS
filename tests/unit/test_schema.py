import pytest
from pydantic import ValidationError

from forgelens.schema import ForgeLensOutput


def test_output_accepts_uncertain_manual_review() -> None:
    output = ForgeLensOutput(
        verdict="uncertain",
        calibrated_risk=0.5,
        tamper_type="unknown",
        affected_fields=[],
        evidence_regions=[],
        tamper_mask_path="",
        recommended_action="manual_review",
        limitations=["Insufficient evidence"],
    )
    assert output.verdict == "uncertain"


def test_output_rejects_invalid_risk() -> None:
    with pytest.raises(ValidationError):
        ForgeLensOutput(
            verdict="forged",
            calibrated_risk=1.1,
            tamper_type="unknown",
            affected_fields=[],
            evidence_regions=[],
            tamper_mask_path="mask.png",
            recommended_action="reject",
            limitations=[],
        )

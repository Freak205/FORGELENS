from pathlib import Path

import torch

from forgelens.calibration import OperatingPolicy
from forgelens.inference import infer_tensor
from forgelens.models import TinyJointDetector


def test_inference_writes_mask_and_validates_schema(tmp_path: Path) -> None:
    model = TinyJointDetector(base_channels=2)
    policy = OperatingPolicy(accept_below=0.0, reject_at_or_above=1.0)
    mask_path = tmp_path / "mask.png"
    output = infer_tensor(
        model,
        torch.rand(3, 32, 48),
        mask_path,
        policy,
    )
    assert output.verdict == "uncertain"
    assert output.recommended_action == "manual_review"
    assert 0.0 <= output.calibrated_risk <= 1.0
    assert mask_path.is_file()
    assert "not forensic proof" in output.limitations[0].lower()

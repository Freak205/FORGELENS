"""Strict, uncertainty-aware inference assembly."""

from pathlib import Path

import torch
from PIL import Image
from torch import Tensor

from forgelens.calibration import OperatingPolicy, TemperatureScaler
from forgelens.models import TinyJointDetector
from forgelens.schema import EvidenceRegion, ForgeLensOutput


def _mask_box(binary_mask: Tensor) -> tuple[int, int, int, int] | None:
    locations = torch.nonzero(binary_mask, as_tuple=False)
    if locations.numel() == 0:
        return None
    y_min, x_min = locations.min(dim=0).values.tolist()
    y_max, x_max = locations.max(dim=0).values.tolist()
    return int(x_min), int(y_min), int(x_max + 1), int(y_max + 1)


@torch.no_grad()
def infer_tensor(
    model: TinyJointDetector,
    image: Tensor,
    output_mask_path: Path,
    policy: OperatingPolicy,
    temperature_scaler: TemperatureScaler | None = None,
    mask_threshold: float = 0.5,
) -> ForgeLensOutput:
    """Run one normalized RGB tensor through the strict output contract."""
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("image must have shape [3, height, width]")
    if not 0.0 < mask_threshold < 1.0:
        raise ValueError("mask threshold must be between zero and one")
    device = next(model.parameters()).device
    model.eval()
    output = model(image.unsqueeze(0).to(device))
    logits = output.image_logits
    if temperature_scaler is not None:
        temperature_scaler = temperature_scaler.to(device).eval()
        logits = temperature_scaler(logits)
    calibrated_risk = float(logits.sigmoid().item())
    verdict, action = policy.decide(calibrated_risk)
    mask_probability = output.mask_logits.sigmoid()[0, 0].cpu()
    binary_mask = mask_probability >= mask_threshold
    output_mask_path.parent.mkdir(parents=True, exist_ok=True)
    mask_image = Image.fromarray((binary_mask.numpy() * 255).astype("uint8"))
    mask_image.save(output_mask_path)
    box = _mask_box(binary_mask)
    evidence_regions = (
        [
            EvidenceRegion(
                box=box,
                observation=(
                    "The localization head exceeded its configured threshold "
                    "in this region; this is model evidence, not forensic proof."
                ),
            )
        ]
        if box is not None
        else []
    )
    limitations = [
        "Research prototype; not forensic proof.",
        "No OCR field attribution is available in this baseline.",
    ]
    if temperature_scaler is None:
        limitations.append("Risk is not temperature-scaled.")
    return ForgeLensOutput(
        verdict=verdict,
        calibrated_risk=calibrated_risk,
        tamper_type="unknown",
        affected_fields=[],
        evidence_regions=evidence_regions,
        tamper_mask_path=str(output_mask_path),
        recommended_action=action,
        limitations=limitations,
    )

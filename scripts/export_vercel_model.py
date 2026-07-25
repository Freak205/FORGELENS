"""Export and verify the deployment-only ONNX inference artifact."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort  # type: ignore[import-untyped]
import torch
from torch import Tensor, nn

from forgelens.models import ResidualUNetJointDetector

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = PROJECT_ROOT / "artifacts/experiments/RESIDUAL-COPYMOVE-001/best.pt"
OUTPUT = PROJECT_ROOT / "api/model.onnx"


class DeploymentModel(nn.Module):
    """Expose the detector's two tensors as stable ONNX outputs."""

    def __init__(self, model: ResidualUNetJointDetector) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        output = self.model(images)
        return output.image_logits, output.mask_logits


def load_model() -> DeploymentModel:
    payload: dict[str, Any] = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )
    model = ResidualUNetJointDetector(base_channels=8)
    model.load_state_dict(payload["model"])
    return DeploymentModel(model.eval()).eval()


def main() -> None:
    """Export a fixed-shape model and compare ONNX Runtime with PyTorch."""
    torch.manual_seed(20250823)
    model = load_model()
    sample = torch.rand(1, 3, 192, 288)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.with_suffix(".onnx.data").unlink(missing_ok=True)
    torch.onnx.export(
        model,
        (sample,),
        OUTPUT,
        export_params=True,
        input_names=["images"],
        output_names=["image_logits", "mask_logits"],
        opset_version=18,
        external_data=False,
    )
    onnx.checker.check_model(onnx.load(OUTPUT))
    with torch.no_grad():
        expected = model(sample)
    session = ort.InferenceSession(
        OUTPUT.as_posix(),
        providers=["CPUExecutionProvider"],
    )
    actual = session.run(None, {"images": sample.numpy()})
    np.testing.assert_allclose(
        actual[0],
        expected[0].numpy(),
        rtol=1e-4,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        actual[1],
        expected[1].numpy(),
        rtol=1e-4,
        atol=1e-5,
    )
    print(f"Exported and verified {OUTPUT} ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

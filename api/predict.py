"""Vercel Function for the safe ForgeLens research demonstration."""

from __future__ import annotations

import base64
import io
import json
import math
import tempfile
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, cast

import torch
from PIL import Image, UnidentifiedImageError
from torchvision.transforms import functional  # type: ignore[import-untyped]

from forgelens.calibration import OperatingPolicy, TemperatureScaler
from forgelens.inference import infer_tensor
from forgelens.models import ResidualUNetJointDetector

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = PROJECT_ROOT / "artifacts/experiments/RESIDUAL-COPYMOVE-001/best.pt"
MAX_UPLOAD_BYTES = 4 * 1024 * 1024
DEVICE = torch.device("cpu")


def load_model() -> ResidualUNetJointDetector:
    """Load the rejected proxy baseline once per warm function instance."""
    payload: dict[str, Any] = torch.load(
        CHECKPOINT, map_location="cpu", weights_only=True
    )
    model = ResidualUNetJointDetector(base_channels=8)
    model.load_state_dict(payload["model"])
    return model.to(DEVICE).eval()


MODEL = load_model()
TEMPERATURE = TemperatureScaler()
TEMPERATURE.log_temperature.data.fill_(math.log(1.0000354051589966))
POLICY = OperatingPolicy(accept_below=0.25, reject_at_or_above=0.75)


def decode_image(payload: bytes) -> torch.Tensor:
    """Validate and normalize a supported document image."""
    try:
        with Image.open(io.BytesIO(payload)) as image_file:
            image_file.verify()
        with Image.open(io.BytesIO(payload)) as image_file:
            image = functional.pil_to_tensor(image_file.convert("RGB")).float() / 255
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("upload is not a valid supported image") from error
    return cast(torch.Tensor, functional.resize(image, (192, 288), antialias=True))


class handler(BaseHTTPRequestHandler):  # noqa: N801
    """Accept a raw image and return a structured research-only prediction."""

    def _respond(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > MAX_UPLOAD_BYTES:
            self._respond(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                {"error": "image must be between 1 byte and 4 MiB"},
            )
            return
        if not self.headers.get("Content-Type", "").startswith("image/"):
            self._respond(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                {"error": "PNG, JPEG, or WebP image required"},
            )
            return

        try:
            image = decode_image(self.rfile.read(length))
            mask_path = (
                Path(tempfile.gettempdir()) / f"forgelens-{uuid.uuid4().hex}.png"
            )
            started = time.perf_counter()
            try:
                output = infer_tensor(
                    MODEL,
                    image,
                    mask_path,
                    POLICY,
                    TEMPERATURE,
                    mask_threshold=0.32,
                ).model_dump(mode="json")
                encoded_mask = base64.b64encode(mask_path.read_bytes()).decode("ascii")
            finally:
                mask_path.unlink(missing_ok=True)
            output.update(
                {
                    "mask_url": f"data:image/png;base64,{encoded_mask}",
                    "latency_ms": (time.perf_counter() - started) * 1000,
                    "model_status": "rejected proxy baseline; manual review only",
                }
            )
            self._respond(HTTPStatus.OK, output)
        except ValueError as error:
            self._respond(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except Exception:
            self._respond(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": "inference is temporarily unavailable"},
            )

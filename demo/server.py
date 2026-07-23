"""Dependency-light local ForgeLens research demo."""

from __future__ import annotations

import io
import json
import math
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
MASK_ROOT = PROJECT_ROOT / "artifacts/demo/masks"
INDEX = Path(__file__).with_name("index.html")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model() -> ResidualUNetJointDetector:
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {CHECKPOINT}")
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
    try:
        with Image.open(io.BytesIO(payload)) as image_file:
            image_file.verify()
        with Image.open(io.BytesIO(payload)) as image_file:
            image = functional.pil_to_tensor(image_file.convert("RGB")).float() / 255
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("upload is not a valid supported image") from error
    return cast(torch.Tensor, functional.resize(image, (192, 288), antialias=True))


class DemoHandler(BaseHTTPRequestHandler):
    """Serve a localhost UI and raw-image prediction endpoint."""

    def _respond(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/":
            self._respond(HTTPStatus.OK, INDEX.read_bytes(), "text/html; charset=utf-8")
            return
        if self.path.startswith("/mask/"):
            mask = MASK_ROOT / Path(self.path.removeprefix("/mask/")).name
            if mask.is_file() and mask.suffix == ".png":
                self._respond(HTTPStatus.OK, mask.read_bytes(), "image/png")
                return
        self._respond(HTTPStatus.NOT_FOUND, b"Not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/predict":
            self._respond(HTTPStatus.NOT_FOUND, b"Not found", "text/plain")
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > MAX_UPLOAD_BYTES:
            self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "invalid upload size")
            return
        if not self.headers.get("Content-Type", "").startswith("image/"):
            self._error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "image upload required")
            return
        try:
            image = decode_image(self.rfile.read(length))
            mask_name = f"{uuid.uuid4().hex}.png"
            started = time.perf_counter()
            output = infer_tensor(
                MODEL,
                image,
                MASK_ROOT / mask_name,
                POLICY,
                TEMPERATURE,
                mask_threshold=0.32,
            ).model_dump(mode="json")
            output.update(
                {
                    "mask_url": f"/mask/{mask_name}",
                    "latency_ms": (time.perf_counter() - started) * 1000,
                    "model_status": "rejected proxy baseline; manual review only",
                }
            )
            self._respond(
                HTTPStatus.OK, json.dumps(output).encode(), "application/json"
            )
        except ValueError as error:
            self._error(HTTPStatus.BAD_REQUEST, str(error))

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._respond(
            status, json.dumps({"error": message}).encode(), "application/json"
        )


def main() -> None:
    MASK_ROOT.mkdir(parents=True, exist_ok=True)
    print("ForgeLens demo: http://127.0.0.1:7860")
    ThreadingHTTPServer(("127.0.0.1", 7860), DemoHandler).serve_forever()


if __name__ == "__main__":
    main()

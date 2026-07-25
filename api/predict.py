"""Vercel Function for the safe ForgeLens research demonstration."""

from __future__ import annotations

import base64
import io
import json
import tempfile
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "api/model.onnx"
MAX_UPLOAD_BYTES = 4 * 1024 * 1024
INPUT_HEIGHT = 192
INPUT_WIDTH = 288
MASK_THRESHOLD = 0.32
TEMPERATURE = 1.0000354051589966
ACCEPT_BELOW = 0.25
REJECT_AT_OR_ABOVE = 0.75
EVIDENCE_OBSERVATION = (
    "The localization head exceeded its configured threshold in this region; "
    "this is model evidence, not forensic proof."
)


def load_model() -> ort.InferenceSession:
    """Load the exported baseline once per warm function instance."""
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        MODEL_PATH.as_posix(),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


SESSION = load_model()


def decode_image(payload: bytes) -> np.ndarray:
    """Validate and normalize a supported document image."""
    try:
        with Image.open(io.BytesIO(payload)) as image_file:
            image_file.verify()
        with Image.open(io.BytesIO(payload)) as image_file:
            rgb = image_file.convert("RGB")
            resized = rgb.resize(
                (INPUT_WIDTH, INPUT_HEIGHT),
                resample=Image.Resampling.BILINEAR,
            )
            image = np.asarray(resized, dtype=np.float32) / np.float32(255.0)
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("upload is not a valid supported image") from error
    return np.ascontiguousarray(image.transpose(2, 0, 1)[None])


def _sigmoid(value: np.ndarray | float) -> np.ndarray | float:
    """Return a numerically stable logistic transform."""
    array = np.asarray(value)
    positive = array >= 0
    result = np.empty_like(array, dtype=np.float32)
    result[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    exponential = np.exp(array[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return float(result) if result.ndim == 0 else result


def _decision(calibrated_risk: float) -> tuple[str, str]:
    if calibrated_risk < ACCEPT_BELOW:
        return "authentic", "accept"
    if calibrated_risk >= REJECT_AT_OR_ABOVE:
        return "forged", "reject"
    return "uncertain", "manual_review"


def _mask_box(binary_mask: np.ndarray) -> tuple[int, int, int, int] | None:
    locations = np.argwhere(binary_mask)
    if locations.size == 0:
        return None
    y_min, x_min = locations.min(axis=0)
    y_max, x_max = locations.max(axis=0)
    return int(x_min), int(y_min), int(x_max + 1), int(y_max + 1)


def infer_image(image: np.ndarray, output_mask_path: Path) -> dict[str, Any]:
    """Run the exported model while preserving the public response contract."""
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError("image must have shape [1, 3, height, width]")
    image_logits, mask_logits = SESSION.run(
        ["image_logits", "mask_logits"],
        {"images": image},
    )
    calibrated_logit = float(image_logits.reshape(-1)[0]) / TEMPERATURE
    calibrated_risk = float(_sigmoid(calibrated_logit))
    verdict, action = _decision(calibrated_risk)
    mask_probability = np.asarray(_sigmoid(mask_logits[0, 0]))
    binary_mask = mask_probability >= MASK_THRESHOLD
    output_mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(binary_mask.astype(np.uint8) * 255).save(output_mask_path)
    box = _mask_box(binary_mask)
    evidence_regions = (
        [{"box": box, "observation": EVIDENCE_OBSERVATION}] if box is not None else []
    )
    return {
        "verdict": verdict,
        "calibrated_risk": calibrated_risk,
        "tamper_type": "unknown",
        "affected_fields": [],
        "evidence_regions": evidence_regions,
        "tamper_mask_path": str(output_mask_path),
        "recommended_action": action,
        "limitations": [
            "Research prototype; not forensic proof.",
            "No OCR field attribution is available in this baseline.",
        ],
    }


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
                output = infer_image(image, mask_path)
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

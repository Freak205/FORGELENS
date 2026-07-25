"""Deployment API contract checks for the lightweight ONNX runtime."""

from __future__ import annotations

import importlib.util
import io
import json
import threading
import tomllib
from http.client import HTTPConnection
from http.server import HTTPServer
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
from PIL import Image

from forgelens.models import ResidualUNetJointDetector

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_api() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "forgelens_vercel_api",
        PROJECT_ROOT / "api/predict.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_vercel_api_preserves_inference_contract(tmp_path: Path) -> None:
    api = _load_api()
    source = Image.fromarray(
        np.full((96, 144, 3), fill_value=127, dtype=np.uint8),
        mode="RGB",
    )
    payload = io.BytesIO()
    source.save(payload, format="PNG")

    image = api.decode_image(payload.getvalue())
    mask_path = tmp_path / "mask.png"
    output = api.infer_image(image, mask_path)

    assert image.shape == (1, 3, 192, 288)
    assert mask_path.is_file()
    assert set(output) == {
        "verdict",
        "calibrated_risk",
        "tamper_type",
        "affected_fields",
        "evidence_regions",
        "tamper_mask_path",
        "recommended_action",
        "limitations",
    }
    assert output["verdict"] in {"authentic", "forged", "uncertain"}
    assert 0.0 <= output["calibrated_risk"] <= 1.0
    assert output["tamper_type"] == "unknown"
    assert output["recommended_action"] in {"accept", "reject", "manual_review"}


def test_vercel_api_rejects_invalid_image() -> None:
    api = _load_api()
    try:
        api.decode_image(b"not an image")
    except ValueError as error:
        assert str(error) == "upload is not a valid supported image"
    else:
        raise AssertionError("invalid image was accepted")


def test_vercel_onnx_export_matches_checkpoint() -> None:
    api = _load_api()
    generator = np.random.default_rng(20250823)
    image = generator.random((1, 3, 192, 288), dtype=np.float32)
    payload = torch.load(
        PROJECT_ROOT / "artifacts/experiments/RESIDUAL-COPYMOVE-001/best.pt",
        map_location="cpu",
        weights_only=True,
    )
    model = ResidualUNetJointDetector(base_channels=8)
    model.load_state_dict(payload["model"])
    model.eval()

    with torch.no_grad():
        expected = model(torch.from_numpy(image))
    actual = api.SESSION.run(
        ["image_logits", "mask_logits"],
        {"images": image},
    )

    np.testing.assert_allclose(
        actual[0],
        expected.image_logits.numpy(),
        rtol=1e-4,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        actual[1],
        expected.mask_logits.numpy(),
        rtol=1e-4,
        atol=1e-5,
    )


def test_vercel_http_handler_returns_existing_api_contract() -> None:
    api = _load_api()
    source = Image.fromarray(
        np.full((48, 72, 3), fill_value=127, dtype=np.uint8),
        mode="RGB",
    )
    payload = io.BytesIO()
    source.save(payload, format="PNG")
    body = payload.getvalue()
    server = HTTPServer(("127.0.0.1", 0), api.handler)
    thread = threading.Thread(target=server.handle_request)
    thread.start()
    try:
        connection = HTTPConnection(*server.server_address, timeout=30)
        connection.request(
            "POST",
            "/api/predict",
            body=body,
            headers={
                "Content-Type": "image/png",
                "Content-Length": str(len(body)),
            },
        )
        response = connection.getresponse()
        output = json.loads(response.read())
        connection.close()
    finally:
        thread.join(timeout=30)
        server.server_close()

    assert response.status == 200
    assert output["mask_url"].startswith("data:image/png;base64,")
    assert output["model_status"] == ("rejected proxy baseline; manual review only")
    assert output["latency_ms"] >= 0.0


def test_vercel_dependency_and_file_trace_stay_inference_only() -> None:
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    dependencies = set(project["project"]["dependencies"])
    assert dependencies == {
        "numpy==2.3.2",
        "onnxruntime==1.27.0",
        "Pillow==12.3.0",
    }
    assert all("torch" not in dependency for dependency in dependencies)

    config = json.loads((PROJECT_ROOT / "vercel.json").read_text())
    function = config["functions"]["api/predict.py"]
    assert function["includeFiles"] == "api/model.onnx"
    assert len(function["excludeFiles"]) <= 256
    assert (PROJECT_ROOT / function["includeFiles"]).is_file()

"""Measure batch-one model latency and memory on a real CORD test image."""

from __future__ import annotations

import hashlib
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from forgelens.data import ManifestDocumentDataset
from forgelens.models import ResidualUNetJointDetector

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STORAGE_ROOT = PROJECT_ROOT.parent
CHECKPOINT = (
    PROJECT_ROOT / "artifacts" / "experiments" / "RESIDUAL-COPYMOVE-001" / "best.pt"
)
OUTPUT = PROJECT_ROOT / "reports" / "tables" / "inference_benchmark.json"
WARMUP = 20
ITERATIONS = 100


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "UNCOMMITTED"


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(fraction * len(ordered)))
    return ordered[index]


def benchmark(
    model: ResidualUNetJointDetector,
    image: torch.Tensor,
    device: torch.device,
    *,
    mixed_precision: bool,
) -> dict[str, float]:
    for _ in range(WARMUP):
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=device.type,
                enabled=mixed_precision,
            ),
        ):
            model(image)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    durations: list[float] = []
    for _ in range(ITERATIONS):
        started = time.perf_counter()
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=device.type,
                enabled=mixed_precision,
            ),
        ):
            model(image)
        if device.type == "cuda":
            torch.cuda.synchronize()
        durations.append((time.perf_counter() - started) * 1000.0)
    median = statistics.median(durations)
    return {
        "median_latency_ms": median,
        "p95_latency_ms": percentile(durations, 0.95),
        "throughput_images_per_second": 1000.0 / median,
        "peak_vram_mb": (
            torch.cuda.max_memory_allocated() / 1048576
            if device.type == "cuda"
            else 0.0
        ),
    }


def main() -> None:
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload: dict[str, Any] = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )
    model = ResidualUNetJointDetector(base_channels=8)
    model.load_state_dict(payload["model"])
    model = model.to(device).eval()
    dataset = ManifestDocumentDataset(
        PROJECT_ROOT / "configs" / "data" / "cord_copy_move_v1.json",
        STORAGE_ROOT,
        "test",
        (192, 288),
    )
    image = dataset[0].image.unsqueeze(0).to(device)
    record = {
        "benchmark_id": "LATENCY-RESIDUAL-001",
        "git_commit": git_commit(),
        "checkpoint_sha256": file_sha256(CHECKPOINT),
        "sample_id": dataset[0].sample_id,
        "input_shape": list(image.shape),
        "warmup_iterations": WARMUP,
        "measured_iterations": ITERATIONS,
        "hardware": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
        "fp32": benchmark(model, image, device, mixed_precision=False),
        "amp_fp16": benchmark(
            model,
            image,
            device,
            mixed_precision=device.type == "cuda",
        ),
        "scope": "model-forward only; excludes file decode and preprocessing",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

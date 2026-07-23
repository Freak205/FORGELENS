"""Strict interoperability helpers for the published TruFor baseline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TruForOutput:
    """Normalized output produced by the official TruFor inference program."""

    score: float
    anomaly_map: NDArray[np.float32]
    confidence_map: NDArray[np.float32]
    image_size: tuple[int, int]


def build_trufor_input_manifest(
    source_manifest: Path,
    dataset_root: Path,
    destination: Path,
    *,
    split: str,
) -> int:
    """Create an immutable input index without copying source images."""
    payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    selected: list[dict[str, Any]] = []
    for item in payload["items"]:
        if item["split"] != split:
            continue
        image_path = (dataset_root / item["relative_path"]).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        selected.append(
            {
                "sample_id": item["sample_id"],
                "image_path": str(image_path),
                "sha256": item["sha256"],
                "label": int(item["label"]),
            }
        )
    if not selected:
        raise ValueError(f"manifest contains no items for split {split!r}")
    output = {
        "schema_version": 1,
        "source_dataset": payload["dataset"],
        "source_revision": payload["revision"],
        "split": split,
        "count": len(selected),
        "items": selected,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return len(selected)


def load_trufor_output(path: Path) -> TruForOutput:
    """Load an official ``.npz`` result and reject malformed values."""
    with np.load(path, allow_pickle=False) as archive:
        required = {"score", "map", "conf", "imgsize"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"TruFor output is missing keys: {sorted(missing)}")
        score = float(np.asarray(archive["score"]).item())
        anomaly_map = np.asarray(archive["map"], dtype=np.float32)
        confidence_map = np.asarray(archive["conf"], dtype=np.float32)
        size_values = np.asarray(archive["imgsize"]).reshape(-1)

    if not 0.0 <= score <= 1.0:
        raise ValueError("TruFor score must be in [0, 1]")
    if anomaly_map.ndim != 2 or confidence_map.shape != anomaly_map.shape:
        raise ValueError("TruFor map and confidence map must be matching 2-D arrays")
    if size_values.size != 2:
        raise ValueError("TruFor imgsize must contain height and width")
    image_size = (int(size_values[0]), int(size_values[1]))
    if image_size != anomaly_map.shape:
        raise ValueError("TruFor imgsize does not match the localization map")
    if not np.isfinite(anomaly_map).all() or not np.isfinite(confidence_map).all():
        raise ValueError("TruFor output contains non-finite map values")
    if not np.logical_and(anomaly_map >= 0.0, anomaly_map <= 1.0).all():
        raise ValueError("TruFor anomaly map must be in [0, 1]")
    if not np.logical_and(confidence_map >= 0.0, confidence_map <= 1.0).all():
        raise ValueError("TruFor confidence map must be in [0, 1]")
    return TruForOutput(
        score=score,
        anomaly_map=anomaly_map,
        confidence_map=confidence_map,
        image_size=image_size,
    )

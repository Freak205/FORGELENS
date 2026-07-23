from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from forgelens.baselines.trufor import (
    build_trufor_input_manifest,
    load_trufor_output,
)


def test_build_trufor_input_manifest(tmp_path: Path) -> None:
    image = tmp_path / "images" / "sample.png"
    image.parent.mkdir()
    image.write_bytes(b"image")
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "dataset": "fixture",
                "revision": "abc",
                "items": [
                    {
                        "sample_id": "one",
                        "relative_path": "images/sample.png",
                        "sha256": "deadbeef",
                        "label": 0,
                        "split": "test",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "out" / "inputs.json"
    assert build_trufor_input_manifest(source, tmp_path, output, split="test") == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["count"] == 1
    assert payload["items"][0]["image_path"] == str(image.resolve())


def test_load_trufor_output(tmp_path: Path) -> None:
    output = tmp_path / "result.npz"
    np.savez(
        output,
        score=np.array(0.75),
        map=np.full((2, 3), 0.25),
        conf=np.full((2, 3), 0.9),
        imgsize=np.array([2, 3]),
    )
    result = load_trufor_output(output)
    assert result.score == pytest.approx(0.75)
    assert result.image_size == (2, 3)


def test_load_trufor_output_rejects_malformed_map(tmp_path: Path) -> None:
    output = tmp_path / "result.npz"
    np.savez(
        output,
        score=np.array(0.75),
        map=np.full((2, 3), 1.25),
        conf=np.full((2, 3), 0.9),
        imgsize=np.array([2, 3]),
    )
    with pytest.raises(ValueError, match="anomaly map"):
        load_trufor_output(output)

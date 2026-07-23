import json
from pathlib import Path

import pytest
from PIL import Image

from forgelens.data import CordAuthenticDataset


def test_cord_adapter_loads_authentic_sample(tmp_path: Path) -> None:
    split_root = tmp_path / "train"
    images = split_root / "images"
    images.mkdir(parents=True)
    Image.new("RGB", (20, 10), "white").save(images / "000001.png")
    record = {
        "sample_id": "cord-train-1",
        "source_group": "cord:train:1",
        "image_path": "images/000001.png",
        "sha256": "fixture",
    }
    (split_root / "metadata.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )
    dataset = CordAuthenticDataset(tmp_path, "train", (32, 48))
    sample = dataset[0]
    assert sample.image.shape == (3, 32, 48)
    assert sample.mask.shape == (1, 32, 48)
    assert sample.mask.sum() == 0
    assert sample.label.item() == 0.0


def test_cord_adapter_rejects_unknown_split(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        CordAuthenticDataset(tmp_path, "other", (32, 48))

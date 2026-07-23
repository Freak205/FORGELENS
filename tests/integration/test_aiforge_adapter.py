import json
from pathlib import Path

import pytest
from PIL import Image

from forgelens.data import AIForgeDocForgeryDataset


def test_aiforge_adapter_loads_and_groups(tmp_path: Path) -> None:
    images = tmp_path / "TrainingSet" / "images"
    masks = tmp_path / "TrainingSet" / "masks"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)
    Image.new("RGB", (20, 10), "white").save(images / "000000001.png")
    mask = Image.new("L", (20, 10), 0)
    for x in range(4, 8):
        for y in range(3, 6):
            mask.putpixel((x, y), 255)
    mask.save(masks / "000000001.png")
    metadata = {
        "new_id": "000000001",
        "image_id": "source-7",
        "source_dataset": "cord",
        "split": "training",
    }
    (tmp_path / "metadata.jsonl").write_text(
        json.dumps(metadata) + "\n", encoding="utf-8"
    )
    dataset = AIForgeDocForgeryDataset(
        tmp_path,
        split="TrainingSet",
        image_size=(32, 48),
        allowed_sources={"cord"},
    )
    sample = dataset[0]
    assert sample.image.shape == (3, 32, 48)
    assert sample.mask.shape == (1, 32, 48)
    assert sample.mask.sum() > 0
    assert sample.label.item() == 1.0
    assert sample.source_group == "cord:source-7"


def test_aiforge_adapter_rejects_missing_metadata(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        AIForgeDocForgeryDataset(tmp_path, "TrainingSet", (32, 48))

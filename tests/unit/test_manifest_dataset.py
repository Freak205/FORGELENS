from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from forgelens.data.manifest import ManifestDocumentDataset


def test_manifest_dataset_loads_authentic_and_forged(tmp_path: Path) -> None:
    image = tmp_path / "data" / "image.png"
    mask = tmp_path / "data" / "mask.png"
    image.parent.mkdir()
    Image.new("RGB", (12, 10), "white").save(image)
    Image.new("L", (12, 10), 255).save(mask)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "sample_id": "auth",
                        "source_group": "group",
                        "split": "train",
                        "image_path": "data/image.png",
                        "mask_path": None,
                        "label": 0,
                    },
                    {
                        "sample_id": "fake",
                        "source_group": "group",
                        "split": "train",
                        "image_path": "data/image.png",
                        "mask_path": "data/mask.png",
                        "label": 1,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    dataset = ManifestDocumentDataset(manifest, tmp_path, "train", (8, 8))
    assert len(dataset) == 2
    assert torch.count_nonzero(dataset[0].mask) == 0
    assert torch.all(dataset[1].mask == 1)
    assert dataset[0].source_group == dataset[1].source_group


def test_manifest_dataset_rejects_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.png"
    outside.write_bytes(b"x")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "sample_id": "bad",
                        "source_group": "bad",
                        "split": "test",
                        "image_path": "../outside.png",
                        "mask_path": None,
                        "label": 0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="escapes"):
        ManifestDocumentDataset(manifest, tmp_path, "test", (8, 8))

"""Adapter for the locally extracted, revision-pinned CORD v2 corpus."""

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode  # type: ignore[import-untyped]
from torchvision.transforms import functional

from forgelens.data.types import DocumentSample


class CordAuthenticDataset(Dataset[DocumentSample]):
    """Load authentic CORD receipts while preserving official splits."""

    def __init__(self, root: Path, split: str, image_size: tuple[int, int]) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        self.split_root = root / split
        manifest = self.split_root / "metadata.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        self.records: list[dict[str, Any]] = [
            json.loads(line)
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not self.records:
            raise ValueError("CORD manifest is empty")
        self.image_size = image_size
        for record in self.records:
            image_path = self.split_root / str(record["image_path"])
            if not image_path.is_file():
                raise FileNotFoundError(image_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> DocumentSample:
        record = self.records[index]
        image_path = self.split_root / str(record["image_path"])
        with Image.open(image_path) as image_file:
            image = functional.pil_to_tensor(image_file.convert("RGB")).float() / 255.0
        image = functional.resize(
            image, self.image_size, interpolation=InterpolationMode.BILINEAR
        )
        mask = torch.zeros((1, *self.image_size), dtype=image.dtype)
        return DocumentSample(
            image=image,
            mask=mask,
            label=torch.tensor(0.0),
            sample_id=str(record["sample_id"]),
            source_group=str(record["source_group"]),
            metadata=record,
        )

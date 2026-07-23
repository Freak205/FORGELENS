"""Manifest-backed dataset for authentic and forged document images."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode  # type: ignore[import-untyped]
from torchvision.transforms import functional

from forgelens.data.types import DocumentSample


class ManifestDocumentDataset(Dataset[DocumentSample]):
    """Load one split from a checksum-carrying ForgeLens manifest."""

    def __init__(
        self,
        manifest_path: Path,
        storage_root: Path,
        split: str,
        image_size: tuple[int, int],
    ) -> None:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.storage_root = storage_root.resolve()
        self.image_size = image_size
        self.records: list[dict[str, Any]] = [
            item for item in payload["items"] if item["split"] == split
        ]
        if not self.records:
            raise ValueError(f"manifest contains no items for split {split!r}")
        for record in self.records:
            self._resolve(record["image_path"])
            mask_path = record.get("mask_path")
            if mask_path is not None:
                self._resolve(mask_path)

    def _resolve(self, relative_path: str) -> Path:
        path = (self.storage_root / relative_path).resolve()
        if not path.is_relative_to(self.storage_root):
            raise ValueError("manifest path escapes storage root")
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> DocumentSample:
        record = self.records[index]
        with Image.open(self._resolve(record["image_path"])) as image_file:
            image = functional.pil_to_tensor(image_file.convert("RGB")).float() / 255.0
        image = functional.resize(
            image,
            self.image_size,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        mask_path = record.get("mask_path")
        if mask_path is None:
            mask = torch.zeros((1, *self.image_size), dtype=torch.float32)
        else:
            with Image.open(self._resolve(mask_path)) as mask_file:
                mask = functional.pil_to_tensor(mask_file.convert("L")).float() / 255.0
            mask = functional.resize(
                mask,
                self.image_size,
                interpolation=InterpolationMode.NEAREST,
            )
            mask = (mask > 0.5).float()
        return DocumentSample(
            image=image,
            mask=mask,
            label=torch.tensor(float(record["label"])),
            sample_id=str(record["sample_id"]),
            source_group=str(record["source_group"]),
            metadata={
                key: value
                for key, value in record.items()
                if key not in {"image_path", "mask_path"}
            },
        )

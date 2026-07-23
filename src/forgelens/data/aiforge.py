"""AIForge-Doc v1/v2 adapter with strict provenance checks."""

import json
from collections.abc import Collection
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode  # type: ignore[import-untyped]
from torchvision.transforms import functional

from forgelens.data.types import DocumentSample


class AIForgeDocForgeryDataset(Dataset[DocumentSample]):
    """Load forged images and masks from an authorized AIForge-Doc snapshot.

    AIForge-Doc contains forged samples. This adapter deliberately labels every
    row as forged and must not be used alone for image-level classification.
    Authentic examples must come from source corpora and share the same
    source-image grouping in the split manifest.
    """

    def __init__(
        self,
        root: Path,
        split: str,
        image_size: tuple[int, int],
        allowed_sources: Collection[str] | None = None,
    ) -> None:
        if split not in {"TrainingSet", "TestingSet"}:
            raise ValueError("split must be TrainingSet or TestingSet")
        self.root = root
        self.split = split
        self.image_size = image_size
        self.images_directory = root / split / "images"
        self.masks_directory = root / split / "masks"
        metadata_path = root / "metadata.jsonl"
        for required_path in (
            self.images_directory,
            self.masks_directory,
            metadata_path,
        ):
            if not required_path.exists():
                raise FileNotFoundError(required_path)
        records = [
            json.loads(line)
            for line in metadata_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        normalized_split = split.removesuffix("Set").lower()
        selected: list[dict[str, Any]] = []
        for record in records:
            record_split = str(record.get("split", "")).lower()
            if record_split and record_split not in {
                normalized_split,
                split.lower(),
            }:
                continue
            source = str(record.get("source_dataset", "")).lower()
            if allowed_sources is not None and source not in allowed_sources:
                continue
            sample_id = str(record.get("new_id", record.get("spec_id", "")))
            sample_id = Path(sample_id).stem
            if not sample_id:
                raise ValueError("metadata row lacks new_id/spec_id")
            record["_sample_id"] = sample_id
            selected.append(record)
        self.records = selected
        if not self.records:
            raise ValueError("no metadata rows match the requested split/source filter")
        for record in self.records:
            sample_id = str(record["_sample_id"])
            if not (self.images_directory / f"{sample_id}.png").is_file():
                raise FileNotFoundError(self.images_directory / f"{sample_id}.png")
            if not (self.masks_directory / f"{sample_id}.png").is_file():
                raise FileNotFoundError(self.masks_directory / f"{sample_id}.png")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> DocumentSample:
        record = self.records[index]
        sample_id = str(record["_sample_id"])
        image_path = self.images_directory / f"{sample_id}.png"
        mask_path = self.masks_directory / f"{sample_id}.png"
        with Image.open(image_path) as image_file:
            image = functional.pil_to_tensor(image_file.convert("RGB")).float() / 255.0
        with Image.open(mask_path) as mask_file:
            mask = functional.pil_to_tensor(mask_file.convert("L")).float() / 255.0
        image = functional.resize(
            image, self.image_size, interpolation=InterpolationMode.BILINEAR
        )
        mask = functional.resize(
            mask, self.image_size, interpolation=InterpolationMode.NEAREST
        )
        mask = (mask > 0.5).float()
        source_image = str(record.get("image_id", record.get("spec_id", sample_id)))
        source_dataset = str(record.get("source_dataset", "unknown")).lower()
        return DocumentSample(
            image=image,
            mask=mask,
            label=torch.tensor(1.0),
            sample_id=sample_id,
            source_group=f"{source_dataset}:{source_image}",
            metadata={
                key: value for key, value in record.items() if key != "_sample_id"
            },
        )

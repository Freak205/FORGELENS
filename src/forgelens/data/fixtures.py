"""Safe, fictional fixtures for pipeline tests.

These images are generated from blank tensors and cannot resemble or modify a
real document. They exist only to verify data, mask, training, and checkpoint
plumbing before an approved research dataset is available.
"""

import torch
from torch.utils.data import Dataset

from forgelens.data.types import DocumentSample


class FictionalDocumentFixtures(Dataset[DocumentSample]):
    """Deterministic blank-canvas receipt-like fixtures."""

    def __init__(
        self,
        size: int = 16,
        height: int = 64,
        width: int = 96,
        seed: int = 20260723,
    ) -> None:
        if size < 2 or height < 32 or width < 48:
            raise ValueError("fixture dimensions or size are too small")
        self.size = size
        self.height = height
        self.width = width
        self.seed = seed

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> DocumentSample:
        if index < 0 or index >= self.size:
            raise IndexError(index)
        generator = torch.Generator().manual_seed(self.seed + index)
        image = torch.full((3, self.height, self.width), 0.94)
        image[:, 4:12, 6 : self.width - 6] = torch.tensor([0.15, 0.35, 0.65]).view(
            3, 1, 1
        )
        for row in range(20, self.height - 8, 9):
            line_width = int(
                torch.randint(
                    self.width // 3,
                    self.width - 14,
                    (1,),
                    generator=generator,
                ).item()
            )
            image[:, row : row + 2, 7:line_width] = 0.55

        forged = index % 2 == 1
        mask = torch.zeros((1, self.height, self.width))
        if forged:
            left = self.width // 2 + (index % 5)
            top = self.height // 2 + (index % 3)
            right = min(left + self.width // 4, self.width - 5)
            bottom = min(top + 7, self.height - 5)
            mask[:, top:bottom, left:right] = 1.0
            noise = torch.rand((3, bottom - top, right - left), generator=generator)
            image[:, top:bottom, left:right] = 0.2 + 0.35 * noise

        return DocumentSample(
            image=image,
            mask=mask,
            label=torch.tensor(float(forged)),
            sample_id=f"fictional-{index:04d}",
            source_group=f"blank-template-{index // 4:03d}",
            metadata={"fixture_only": True},
        )

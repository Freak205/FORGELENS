"""Shared dataset sample contracts."""

from dataclasses import dataclass, field
from typing import Any

from torch import Tensor


@dataclass(frozen=True)
class DocumentSample:
    """One document image with detection/localization labels and provenance."""

    image: Tensor
    mask: Tensor
    label: Tensor
    sample_id: str
    source_group: str
    metadata: dict[str, Any] = field(default_factory=dict)

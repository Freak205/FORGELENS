"""Validated inference output schema."""

from typing import Literal

from pydantic import BaseModel, Field


class EvidenceRegion(BaseModel):
    """Localized observation supporting a decision."""

    box: tuple[int, int, int, int]
    observation: str = Field(min_length=1)


class ForgeLensOutput(BaseModel):
    """Strict public output contract."""

    verdict: Literal["authentic", "forged", "uncertain"]
    calibrated_risk: float = Field(ge=0.0, le=1.0)
    tamper_type: Literal["numeric_edit", "text_edit", "photo_replacement", "unknown"]
    affected_fields: list[str]
    evidence_regions: list[EvidenceRegion]
    tamper_mask_path: str
    recommended_action: Literal["accept", "reject", "manual_review"]
    limitations: list[str]

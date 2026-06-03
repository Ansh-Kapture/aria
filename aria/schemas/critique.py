from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class CredibilityScore(BaseModel):
    recency: float = Field(ge=0.0, le=3.0, description="0-3: how recent is the source")
    authority: float = Field(ge=0.0, le=3.0, description="0-3: domain credibility")
    specificity: float = Field(ge=0.0, le=2.0, description="0-2: quantitative vs vague")
    corroboration: float = Field(ge=0.0, le=2.0, description="0-2: supported by other sources")

    @property
    def total(self) -> float:
        return self.recency + self.authority + self.specificity + self.corroboration


class Conflict(BaseModel):
    claim_a: str
    source_a: str
    claim_b: str
    source_b: str
    conflict_type: Literal["factual", "numerical", "temporal", "definitional"]
    similarity_score: float = Field(ge=0.0, le=1.0)


class ConflictResolution(BaseModel):
    conflict: Conflict
    winner: Literal["a", "b", "neither", "both_valid"]
    score_a: CredibilityScore
    score_b: CredibilityScore
    explanation: str
    resolved_claim: str
    unresolved: bool = False
    unresolved_note: Optional[str] = None

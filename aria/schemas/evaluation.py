from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class QualityScore(BaseModel):
    factual_grounding: float = Field(ge=0.0, le=10.0, description="Claims backed by citations")
    coverage: float = Field(ge=0.0, le=10.0, description="Addresses all aspects of sub-question")
    coherence: float = Field(ge=0.0, le=10.0, description="Logical flow, no internal contradictions")
    citation_quality: float = Field(ge=0.0, le=10.0, description="Citations are relevant and accessible")
    aggregate: float = Field(default=0.0, ge=0.0, le=10.0)
    verdict: Literal["pass", "replan", "skip"] = "replan"
    improvement_suggestions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def compute_aggregate(self) -> "QualityScore":
        self.aggregate = (
            self.factual_grounding * 0.35
            + self.coverage * 0.30
            + self.coherence * 0.20
            + self.citation_quality * 0.15
        )
        return self


class EvaluationResult(BaseModel):
    task_id: str
    section_title: str
    score: QualityScore
    passed: bool
    threshold_used: float

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class SearchResult(BaseModel):
    url: str
    title: str
    snippet: str
    full_text: str = ""


class Chunk(BaseModel):
    text: str
    source_url: str
    source_title: str
    relevance_score: float = 0.0
    chunk_id: str
    position: int = 0

    @field_validator("relevance_score", mode="before")
    @classmethod
    def clamp_score(cls, v: float) -> float:
        # Cross-encoder logits are unbounded; normalize silently.
        return max(0.0, min(1.0, float(v)))


class Citation(BaseModel):
    id: str
    url: str
    title: str
    snippet: str


class RetrievalResult(BaseModel):
    task_id: str
    chunks: list[Chunk]
    citations: list[Citation]
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    hyde_hypothesis: str = ""
    iterations: int = 1

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))

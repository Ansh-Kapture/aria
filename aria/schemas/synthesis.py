from __future__ import annotations

from pydantic import BaseModel, Field

from aria.schemas.retrieval import Citation
from aria.schemas.critique import ConflictResolution


class SynthesisResult(BaseModel):
    task_id: str
    section_title: str
    content: str
    citations: list[Citation]
    conflict_resolutions: list[ConflictResolution] = Field(default_factory=list)
    word_count: int = 0

    def model_post_init(self, __context):
        if self.word_count == 0:
            self.word_count = len(self.content.split())


class ReportSection(BaseModel):
    title: str
    content: str
    citations: list[Citation]
    conflict_resolutions: list[ConflictResolution] = Field(default_factory=list)
    quality_scores: dict = Field(default_factory=dict)
    task_id: str = ""

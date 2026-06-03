from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class SubTask(BaseModel):
    id: str
    question: str
    dependencies: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    context_budget: int = 4096
    replan_count: int = 0
    complexity: Literal["simple", "moderate", "complex"] = "moderate"

    model_config = {"use_enum_values": True}


class DAGNode(BaseModel):
    task: SubTask
    children: list[str] = Field(default_factory=list)


class SubTaskList(BaseModel):
    tasks: list[SubTask]
    rationale: str = Field(description="Brief explanation of decomposition strategy")


class ReplanDecision(BaseModel):
    action: Literal["retry", "rephrase", "merge", "skip"]
    reason: str
    failure_dimension: Literal[
        "insufficient_sources", "contradictory_claims", "off_topic", "low_coherence", "max_cycles"
    ]
    revised_question: Optional[str] = None
    merge_with: Optional[str] = None

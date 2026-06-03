from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

from aria.schemas.critique import ConflictResolution
from aria.schemas.retrieval import Chunk
from aria.schemas.synthesis import SynthesisResult
from aria.schemas.evaluation import EvaluationResult

logger = logging.getLogger(__name__)


class SharedWorkingMemory:
    """Thread-safe shared memory accessible to all agents.

    Provides serializable state for crash recovery and redundancy detection.
    All mutations go through an asyncio.Lock to prevent data races under
    concurrent agent execution.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._retrieved_chunks: dict[str, Chunk] = {}       # chunk_id → Chunk
        self._seen_chunk_ids: set[str] = set()              # for O(1) dedup
        self._resolved_conflicts: list[ConflictResolution] = []
        self._completed_sections: dict[str, SynthesisResult] = {}  # task_id → result
        self._evaluation_results: dict[str, EvaluationResult] = {}  # task_id → eval
        self._task_attempts: dict[str, int] = {}            # task_id → attempt count
        self._metadata: dict[str, Any] = {}

    # ── Write operations ────────────────────────────────────────────────────

    async def store_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Store chunks, skipping already-known ones. Returns newly stored chunks."""
        async with self._lock:
            new_chunks = []
            for chunk in chunks:
                if chunk.chunk_id not in self._seen_chunk_ids:
                    self._seen_chunk_ids.add(chunk.chunk_id)
                    self._retrieved_chunks[chunk.chunk_id] = chunk
                    new_chunks.append(chunk)
            return new_chunks

    async def store_conflict_resolution(self, resolution: ConflictResolution) -> None:
        async with self._lock:
            self._resolved_conflicts.append(resolution)

    async def store_section(self, result: SynthesisResult) -> None:
        async with self._lock:
            self._completed_sections[result.task_id] = result

    async def store_evaluation(self, result: EvaluationResult) -> None:
        async with self._lock:
            self._evaluation_results[result.task_id] = result

    async def increment_attempts(self, task_id: str) -> int:
        async with self._lock:
            self._task_attempts[task_id] = self._task_attempts.get(task_id, 0) + 1
            return self._task_attempts[task_id]

    async def set_metadata(self, key: str, value: Any) -> None:
        async with self._lock:
            self._metadata[key] = value

    # ── Read operations (lock-free for non-mutating reads) ──────────────────

    def is_known(self, chunk_id: str) -> bool:
        return chunk_id in self._seen_chunk_ids

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        return self._retrieved_chunks.get(chunk_id)

    def get_section(self, task_id: str) -> Optional[SynthesisResult]:
        return self._completed_sections.get(task_id)

    def get_evaluation(self, task_id: str) -> Optional[EvaluationResult]:
        return self._evaluation_results.get(task_id)

    def get_all_sections(self) -> dict[str, SynthesisResult]:
        return dict(self._completed_sections)

    def get_all_evaluations(self) -> dict[str, EvaluationResult]:
        return dict(self._evaluation_results)

    def get_conflict_resolutions(self) -> list[ConflictResolution]:
        return list(self._resolved_conflicts)

    def get_all_chunks(self) -> list[Chunk]:
        return list(self._retrieved_chunks.values())

    def get_attempts(self, task_id: str) -> int:
        return self._task_attempts.get(task_id, 0)

    def get_snapshot(self) -> dict[str, Any]:
        """Safe read-only view for orchestrator context."""
        return {
            "chunk_count": len(self._retrieved_chunks),
            "completed_tasks": list(self._completed_sections.keys()),
            "evaluation_results": {
                tid: {
                    "aggregate": ev.score.aggregate,
                    "verdict": ev.score.verdict,
                }
                for tid, ev in self._evaluation_results.items()
            },
            "conflict_count": len(self._resolved_conflicts),
            "metadata": deepcopy(self._metadata),
        }

    # ── Persistence ──────────────────────────────────────────────────────────

    async def persist(self, path: str | Path) -> None:
        """Serialize memory to JSON for crash recovery."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        async with self._lock:
            data = {
                "retrieved_chunks": {
                    cid: chunk.model_dump()
                    for cid, chunk in self._retrieved_chunks.items()
                },
                "seen_chunk_ids": list(self._seen_chunk_ids),
                "resolved_conflicts": [r.model_dump() for r in self._resolved_conflicts],
                "completed_sections": {
                    tid: s.model_dump()
                    for tid, s in self._completed_sections.items()
                },
                "evaluation_results": {
                    tid: e.model_dump()
                    for tid, e in self._evaluation_results.items()
                },
                "task_attempts": self._task_attempts,
                "metadata": self._metadata,
            }

        path.write_text(json.dumps(data, indent=2, default=str))
        logger.info("Memory persisted to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "SharedWorkingMemory":
        """Restore memory from a persisted JSON checkpoint."""
        path = Path(path)
        if not path.exists():
            logger.warning("No checkpoint at %s, starting fresh", path)
            return cls()

        data = json.loads(path.read_text())
        memory = cls()

        memory._retrieved_chunks = {
            cid: Chunk(**chunk)
            for cid, chunk in data.get("retrieved_chunks", {}).items()
        }
        memory._seen_chunk_ids = set(data.get("seen_chunk_ids", []))
        memory._resolved_conflicts = [
            ConflictResolution(**r) for r in data.get("resolved_conflicts", [])
        ]
        memory._completed_sections = {
            tid: SynthesisResult(**s)
            for tid, s in data.get("completed_sections", {}).items()
        }
        memory._evaluation_results = {
            tid: EvaluationResult(**e)
            for tid, e in data.get("evaluation_results", {}).items()
        }
        memory._task_attempts = data.get("task_attempts", {})
        memory._metadata = data.get("metadata", {})

        logger.info(
            "Memory loaded from %s: %d chunks, %d sections",
            path,
            len(memory._retrieved_chunks),
            len(memory._completed_sections),
        )
        return memory

    def __repr__(self) -> str:
        return (
            f"SharedWorkingMemory("
            f"chunks={len(self._retrieved_chunks)}, "
            f"sections={len(self._completed_sections)}, "
            f"conflicts={len(self._resolved_conflicts)})"
        )


def make_chunk_id(text: str, url: str) -> str:
    return hashlib.sha256(f"{url}::{text[:200]}".encode()).hexdigest()[:16]

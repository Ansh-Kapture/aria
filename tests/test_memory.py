"""Tests for SharedWorkingMemory: thread safety, deduplication, persistence."""
from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from aria.memory.state import SharedWorkingMemory, make_chunk_id
from aria.schemas.retrieval import Chunk, Citation
from aria.schemas.synthesis import SynthesisResult
from aria.schemas.evaluation import EvaluationResult, QualityScore


def make_chunk(text: str, url: str = "http://example.com", score: float = 0.8) -> Chunk:
    return Chunk(
        text=text,
        source_url=url,
        source_title="Test Source",
        relevance_score=score,
        chunk_id=make_chunk_id(text, url),
    )


def make_synthesis(task_id: str, title: str = "Test Section") -> SynthesisResult:
    return SynthesisResult(
        task_id=task_id,
        section_title=title,
        content="This is test content with [c1] citations.",
        citations=[Citation(id="c1", url="http://example.com", title="Test", snippet="snippet")],
    )


def make_eval(task_id: str, aggregate: float = 7.5) -> EvaluationResult:
    score = QualityScore(
        factual_grounding=aggregate,
        coverage=aggregate,
        coherence=aggregate,
        citation_quality=aggregate,
        verdict="pass" if aggregate >= 6.5 else "replan",
    )
    return EvaluationResult(
        task_id=task_id,
        section_title="Test",
        score=score,
        passed=aggregate >= 6.5,
        threshold_used=6.5,
    )


@pytest.mark.asyncio
async def test_store_and_retrieve_chunks():
    memory = SharedWorkingMemory()
    chunk = make_chunk("Alpha is a transformer limitation", "http://a.com")

    stored = await memory.store_chunks([chunk])
    assert len(stored) == 1
    assert memory.is_known(chunk.chunk_id)
    assert memory.get_chunk(chunk.chunk_id) is not None


@pytest.mark.asyncio
async def test_deduplication():
    memory = SharedWorkingMemory()
    chunk = make_chunk("Duplicate chunk", "http://b.com")

    first = await memory.store_chunks([chunk])
    second = await memory.store_chunks([chunk])  # same chunk again

    assert len(first) == 1
    assert len(second) == 0  # should be deduplicated


@pytest.mark.asyncio
async def test_concurrent_stores_no_race():
    """Concurrent stores must not cause duplicate entries."""
    memory = SharedWorkingMemory()

    chunks = [make_chunk(f"Chunk {i}", f"http://c{i}.com") for i in range(20)]

    async def store_batch(batch):
        return await memory.store_chunks(batch)

    batches = [chunks[i:i+5] for i in range(0, 20, 5)]
    results = await asyncio.gather(*[store_batch(b) for b in batches])

    total_stored = sum(len(r) for r in results)
    assert total_stored == 20, f"Expected 20 unique chunks, got {total_stored}"
    assert len(memory.get_all_chunks()) == 20


@pytest.mark.asyncio
async def test_persist_and_load(tmp_path):
    memory = SharedWorkingMemory()
    chunk = make_chunk("Persisted chunk", "http://persist.com")
    section = make_synthesis("t1")
    eval_result = make_eval("t1")

    await memory.store_chunks([chunk])
    await memory.store_section(section)
    await memory.store_evaluation(eval_result)

    checkpoint = tmp_path / "checkpoint.json"
    await memory.persist(str(checkpoint))
    assert checkpoint.exists()

    restored = SharedWorkingMemory.load(str(checkpoint))
    assert restored.is_known(chunk.chunk_id)
    assert restored.get_section("t1") is not None
    assert restored.get_section("t1").section_title == "Test Section"
    assert restored.get_evaluation("t1") is not None
    assert restored.get_evaluation("t1").score.aggregate == pytest.approx(7.5, abs=0.1)


@pytest.mark.asyncio
async def test_load_nonexistent_file():
    memory = SharedWorkingMemory.load("/nonexistent/path.json")
    assert memory is not None
    assert len(memory.get_all_chunks()) == 0


@pytest.mark.asyncio
async def test_snapshot_is_read_only():
    memory = SharedWorkingMemory()
    await memory.store_section(make_synthesis("t1"))

    snapshot = memory.get_snapshot()
    assert "t1" in snapshot["completed_tasks"]

    # Mutating snapshot should not affect memory
    snapshot["completed_tasks"].append("t_fake")
    assert "t_fake" not in memory.get_snapshot()["completed_tasks"]


@pytest.mark.asyncio
async def test_increment_attempts():
    memory = SharedWorkingMemory()
    count1 = await memory.increment_attempts("t1")
    count2 = await memory.increment_attempts("t1")
    count3 = await memory.increment_attempts("t1")

    assert count1 == 1
    assert count2 == 2
    assert count3 == 3
    assert memory.get_attempts("t1") == 3

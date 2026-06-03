"""Tests for OrchestratorAgent replanning logic and DAG management."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aria.dag.executor import DAGExecutor, build_dag, validate_dag
from aria.memory.state import SharedWorkingMemory
from aria.schemas.evaluation import EvaluationResult, QualityScore
from aria.schemas.synthesis import SynthesisResult
from aria.schemas.retrieval import Citation
from aria.schemas.task import ReplanDecision, SubTask, TaskStatus


def make_subtask(
    id: str,
    question: str = "What is X?",
    deps: list[str] | None = None,
    complexity: str = "moderate",
) -> SubTask:
    return SubTask(
        id=id,
        question=question,
        dependencies=deps or [],
        complexity=complexity,
    )


def make_synthesis(task_id: str) -> SynthesisResult:
    return SynthesisResult(
        task_id=task_id,
        section_title=f"Section {task_id}",
        content=f"Content for {task_id} with [c1] citation.",
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
        section_title=f"Section {task_id}",
        score=score,
        passed=aggregate >= 6.5,
        threshold_used=6.5,
    )


# ── DAG tests ─────────────────────────────────────────────────────────────────

def test_build_dag_no_deps():
    tasks = [make_subtask("t1"), make_subtask("t2"), make_subtask("t3")]
    dag = build_dag(tasks)
    assert len(dag) == 3
    for node in dag.values():
        assert node.task.dependencies == []


def test_build_dag_with_deps():
    tasks = [
        make_subtask("t1"),
        make_subtask("t2", deps=["t1"]),
        make_subtask("t3", deps=["t1", "t2"]),
    ]
    dag = build_dag(tasks)
    assert "t2" in dag["t1"].children
    assert "t3" in dag["t1"].children
    assert "t3" in dag["t2"].children


def test_validate_dag_no_errors():
    tasks = [make_subtask("t1"), make_subtask("t2", deps=["t1"])]
    dag = build_dag(tasks)
    errors = validate_dag(dag)
    assert errors == []


def test_validate_dag_unknown_dep():
    tasks = [make_subtask("t1", deps=["t_unknown"])]
    dag = build_dag(tasks)
    errors = validate_dag(dag)
    assert any("unknown" in e.lower() for e in errors)


def test_validate_dag_cycle():
    tasks = [
        make_subtask("t1", deps=["t2"]),
        make_subtask("t2", deps=["t1"]),
    ]
    dag = build_dag(tasks)
    errors = validate_dag(dag)
    assert len(errors) > 0


# ── DAG executor tests ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_dag_executor_simple():
    """All independent tasks run in parallel and complete."""
    tasks = [make_subtask("t1"), make_subtask("t2"), make_subtask("t3")]
    dag = build_dag(tasks)

    order = []

    async def runner(task: SubTask, dep_ctx: str):
        order.append(task.id)
        return make_synthesis(task.id)

    executor = DAGExecutor(max_parallel=4)
    results = await executor.execute(dag, runner)

    assert set(results.keys()) == {"t1", "t2", "t3"}
    assert set(order) == {"t1", "t2", "t3"}


@pytest.mark.asyncio
async def test_dag_executor_respects_deps():
    """A dependent task only runs after its dependency completes."""
    completed_order = []

    tasks = [
        make_subtask("t1"),
        make_subtask("t2", deps=["t1"]),
    ]
    dag = build_dag(tasks)

    async def runner(task: SubTask, dep_ctx: str):
        completed_order.append(task.id)
        if task.id == "t2":
            assert "t1" in completed_order, "t2 ran before t1"
        return make_synthesis(task.id)

    executor = DAGExecutor(max_parallel=4)
    results = await executor.execute(dag, runner)
    assert list(results.keys()) == ["t1", "t2"] or set(results.keys()) == {"t1", "t2"}
    assert completed_order.index("t1") < completed_order.index("t2")


@pytest.mark.asyncio
async def test_dag_executor_dep_context_passed():
    """Dependency context from t1 is passed to t2."""
    received_context = {}

    tasks = [make_subtask("t1"), make_subtask("t2", deps=["t1"])]
    dag = build_dag(tasks)

    async def runner(task: SubTask, dep_ctx: str):
        received_context[task.id] = dep_ctx
        synthesis = make_synthesis(task.id)
        return synthesis

    executor = DAGExecutor(max_parallel=4)
    await executor.execute(dag, runner)

    assert received_context["t1"] == ""  # no dependencies
    assert "t1" in received_context["t2"]  # context from t1


@pytest.mark.asyncio
async def test_dag_executor_handles_failure():
    """Failed tasks do not block independent tasks."""
    tasks = [make_subtask("t1"), make_subtask("t2")]
    dag = build_dag(tasks)

    async def runner(task: SubTask, dep_ctx: str):
        if task.id == "t1":
            raise RuntimeError("t1 failed")
        return make_synthesis(task.id)

    executor = DAGExecutor(max_parallel=4)
    results = await executor.execute(dag, runner)

    assert "t2" in results
    assert "t1" not in results


# ── Replanning logic tests ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_replan_skips_at_max_cycles():
    from aria.agents.orchestrator import OrchestratorAgent
    from aria.config import Settings

    settings = Settings(anthropic_api_key="test-key", max_replan_cycles=3)
    memory = SharedWorkingMemory()
    orchestrator = OrchestratorAgent(settings=settings, memory=memory)

    task = make_subtask("t1")
    task.replan_count = 3  # at max

    decision = await orchestrator._decide_replan(task)
    assert decision.action == "skip"
    assert decision.failure_dimension == "max_cycles"


@pytest.mark.asyncio
async def test_replan_chooses_retry_for_low_citations():
    from aria.agents.orchestrator import OrchestratorAgent
    from aria.config import Settings

    settings = Settings(anthropic_api_key="test-key", max_replan_cycles=3)
    memory = SharedWorkingMemory()
    orchestrator = OrchestratorAgent(settings=settings, memory=memory)

    task = make_subtask("t1")
    task.replan_count = 1

    # Store an evaluation with low citation quality
    eval_result = EvaluationResult(
        task_id="t1",
        section_title="Test",
        score=QualityScore(
            factual_grounding=7.0,
            coverage=7.0,
            coherence=7.0,
            citation_quality=2.0,  # very low — should trigger retry
            verdict="replan",
        ),
        passed=False,
        threshold_used=6.5,
    )
    await memory.store_evaluation(eval_result)

    decision = await orchestrator._decide_replan(task)
    assert decision.action in ("retry", "rephrase")
    assert decision.failure_dimension == "insufficient_sources"


@pytest.mark.asyncio
async def test_replan_chooses_rephrase_for_low_coverage():
    from aria.agents.orchestrator import OrchestratorAgent
    from aria.config import Settings

    settings = Settings(anthropic_api_key="test-key", max_replan_cycles=3)
    memory = SharedWorkingMemory()
    orchestrator = OrchestratorAgent(settings=settings, memory=memory)

    task = make_subtask("t1")
    task.replan_count = 0

    eval_result = EvaluationResult(
        task_id="t1",
        section_title="Test",
        score=QualityScore(
            factual_grounding=8.0,
            coverage=3.0,  # very low coverage → rephrase
            coherence=8.0,
            citation_quality=7.0,
            verdict="replan",
        ),
        passed=False,
        threshold_used=6.5,
    )
    await memory.store_evaluation(eval_result)

    with patch.object(
        orchestrator, "_rephrase_for_coverage", new_callable=AsyncMock,
        return_value="Rephrased: broader question"
    ):
        decision = await orchestrator._decide_replan(task)

    assert decision.action == "rephrase"
    assert decision.failure_dimension == "off_topic"


def test_quality_score_aggregate_weights():
    """Aggregate = 0.35*factual + 0.30*coverage + 0.20*coherence + 0.15*citation."""
    score = QualityScore(
        factual_grounding=8.0,
        coverage=6.0,
        coherence=7.0,
        citation_quality=5.0,
    )
    expected = 0.35 * 8.0 + 0.30 * 6.0 + 0.20 * 7.0 + 0.15 * 5.0
    assert score.aggregate == pytest.approx(expected, abs=0.01)


def test_quality_score_verdict_pass():
    score = QualityScore(
        factual_grounding=8.0, coverage=8.0, coherence=8.0, citation_quality=8.0,
        verdict="pass",
    )
    assert score.aggregate > 6.5


def test_expand_query_adds_context():
    from aria.agents.orchestrator import OrchestratorAgent
    from aria.config import Settings

    settings = Settings(anthropic_api_key="test-key")
    orch = OrchestratorAgent(settings=settings)
    expanded = orch._expand_query("What is transformer attention?")
    assert "transformer attention" in expanded.lower()
    assert len(expanded) > len("What is transformer attention?")

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import AsyncIterator, Optional

from aria.agents.base import BaseAgent
from aria.agents.citation_verifier import CitationVerifier
from aria.agents.critique import CritiqueAgent
from aria.agents.evaluator import QualityEvaluator
from aria.agents.retrieval import RetrievalAgent
from aria.agents.synthesis import SynthesisAgent
from aria.config import Settings
from aria.dag.executor import DAGExecutor, build_dag, validate_dag
from aria.memory.state import SharedWorkingMemory
from aria.schemas.evaluation import EvaluationResult
from aria.schemas.synthesis import SynthesisResult
from aria.schemas.task import (
    DAGNode,
    ReplanDecision,
    SubTask,
    SubTaskList,
    TaskStatus,
)

logger = logging.getLogger(__name__)

_BUDGET_MAP = {
    "simple": "budget_simple",
    "moderate": "budget_moderate",
    "complex": "budget_complex",
}


class OrchestratorAgent(BaseAgent):
    """Top-level orchestrator: decomposes a query into a DAG, coordinates
    specialist agents, handles dynamic replanning, and enforces termination.

    Design references:
    - ReAct (Yao et al., 2022) for interleaved reasoning/action
    - STORM (Shao et al., 2024) for structured multi-agent outline
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        memory: Optional[SharedWorkingMemory] = None,
        stream: bool = False,
    ) -> None:
        super().__init__(settings=settings)
        self.memory = memory or SharedWorkingMemory()
        self.stream = stream
        self._retrieval = RetrievalAgent(settings=self.settings)
        self._critique = CritiqueAgent(settings=self.settings)
        self._synthesis = SynthesisAgent(settings=self.settings)
        self._evaluator = QualityEvaluator(
            threshold=self.settings.quality_threshold, settings=self.settings
        )
        self._verifier = CitationVerifier(settings=self.settings)
        self._executor = DAGExecutor(max_parallel=self.settings.max_parallel_agents)
        self._stream_callbacks: list = []

    @property
    def system_prompt(self) -> str:
        return (
            "You are a research orchestration system. Given a complex research query, "
            "decompose it into a minimal but complete set of sub-questions that together "
            "answer the full query. Identify explicit dependencies between sub-questions: "
            "mark a sub-question as depending on another only if answering it REQUIRES "
            "the answer to the other first (multi-hop reasoning). "
            "Classify complexity: 'simple' (direct factual), 'moderate' (comparative), "
            "'complex' (synthesizing multiple concepts). Always respond with valid JSON."
        )

    # ── Public API ────────────────────────────────────────────────────────────

    async def run(
        self,
        query: str,
        state_file: Optional[str] = None,
    ) -> dict[str, SynthesisResult]:
        """Orchestrate a full research run for the given query.

        Returns a mapping of task_id → SynthesisResult for all completed tasks.
        """
        start_time = time.monotonic()
        await self.memory.set_metadata("query", query)
        await self.memory.set_metadata("start_time", start_time)

        # Load checkpoint if provided
        if state_file and Path(state_file).exists():
            self.memory = SharedWorkingMemory.load(state_file)
            logger.info("Resuming from checkpoint: %s", state_file)

        # Step 1: Decompose query into DAG
        logger.info("[Orchestrator] Decomposing query: %r", query[:100])
        dag = await self._decompose_query(query)

        if not dag:
            logger.error("[Orchestrator] Failed to decompose query into sub-tasks")
            return {}

        errors = validate_dag(dag)
        if errors:
            for e in errors:
                logger.warning("[Orchestrator] DAG validation: %s", e)

        # Skip tasks that are already complete (resume support)
        completed_snapshots = self.memory.get_all_sections()
        for task_id in completed_snapshots:
            if task_id in dag:
                dag[task_id].task.status = TaskStatus.COMPLETED
                logger.info("[Orchestrator] Skipping already-complete task: %s", task_id)

        logger.info("[Orchestrator] DAG ready: %d tasks", len(dag))
        await self.memory.set_metadata("task_ids", list(dag.keys()))

        # Step 2: Execute DAG with replanning
        results = await self._executor.execute(
            dag,
            task_runner=lambda task, dep_ctx: self._run_task_with_replan(task, dep_ctx),
            on_complete=lambda task: logger.info("[Orchestrator] Task complete: %s", task.id),
        )

        # Merge any already-complete sections from memory
        for task_id, section in self.memory.get_all_sections().items():
            if task_id not in results:
                results[task_id] = section

        # Step 3: Persist final state
        if state_file:
            await self.memory.persist(state_file)

        elapsed = time.monotonic() - start_time
        logger.info(
            "[Orchestrator] Complete: %d sections in %.1fs",
            len(results), elapsed,
        )
        return results

    # ── Task execution + replanning ──────────────────────────────────────────

    async def _run_task_with_replan(
        self, task: SubTask, dependency_context: str
    ) -> Optional[SynthesisResult]:
        """Run a single task through the full pipeline with replanning loop."""
        await self.memory.set_metadata(f"task_{task.id}_status", "in_progress")

        for attempt in range(self.settings.max_replan_cycles + 1):
            task.replan_count = attempt

            if attempt > 0:
                # Apply replan strategy before retrying
                replan = await self._decide_replan(task)
                if replan.action == "skip":
                    logger.warning(
                        "[Orchestrator] Skipping task %s after %d attempts: %s",
                        task.id, attempt, replan.reason,
                    )
                    task.status = TaskStatus.SKIPPED
                    return None
                elif replan.action == "rephrase" and replan.revised_question:
                    logger.info(
                        "[Orchestrator] Rephrasing task %s: %r → %r",
                        task.id, task.question, replan.revised_question,
                    )
                    task.question = replan.revised_question
                elif replan.action == "merge" and replan.merge_with:
                    merged = await self._merge_task(task, replan.merge_with)
                    if merged:
                        return merged

            try:
                result = await self._execute_pipeline(task, dependency_context)
                if result is None:
                    continue

                # Evaluate quality
                eval_result = await self._evaluator.run(task, result)
                await self.memory.store_evaluation(eval_result)

                if eval_result.passed or attempt >= self.settings.max_replan_cycles:
                    await self.memory.store_section(result)
                    if self.stream:
                        self._emit_section(result)
                    return result

                logger.info(
                    "[Orchestrator] Task %s scored %.2f < %.1f (attempt %d/%d)",
                    task.id, eval_result.score.aggregate,
                    self.settings.quality_threshold,
                    attempt + 1, self.settings.max_replan_cycles + 1,
                )
            except Exception as exc:
                logger.error("[Orchestrator] Task %s error (attempt %d): %s", task.id, attempt, exc, exc_info=True)

        # All attempts exhausted — use last result if any
        section = self.memory.get_section(task.id)
        if section:
            return section
        task.status = TaskStatus.FAILED
        return None

    async def _execute_pipeline(
        self, task: SubTask, dependency_context: str
    ) -> Optional[SynthesisResult]:
        """Run the full retrieval → critique → synthesis pipeline for one task."""
        context_budget = getattr(self.settings, _BUDGET_MAP.get(task.complexity, "budget_moderate"))
        task.context_budget = context_budget

        # 1. Retrieval (HyDE + web + rerank)
        retrieval = await self._retrieval.run(task, self.memory)

        # 2. Conflict detection + resolution
        conflicts = await self._critique.run(retrieval.chunks)
        for c in conflicts:
            await self.memory.store_conflict_resolution(c)

        # 3. Synthesis
        synthesis = await self._synthesis.run(
            task, retrieval, conflicts, dependency_context
        )

        # 4. Citation verification (bonus — spot-check)
        verifications = await self._verifier.run(synthesis.citations, synthesis.content)
        unverified = [v for v in verifications if not v.verified]
        if unverified:
            logger.warning(
                "[Orchestrator] Task %s: %d unverified citation(s): %s",
                task.id, len(unverified), [v.citation_id for v in unverified],
            )

        return synthesis

    # ── Replanning logic ─────────────────────────────────────────────────────

    async def _decide_replan(self, task: SubTask) -> ReplanDecision:
        """Diagnose why the task failed and pick a corrective action.

        This is deliberately non-trivial: it inspects the specific quality
        dimension that failed and chooses a targeted response strategy.
        """
        if task.replan_count >= self.settings.max_replan_cycles:
            return ReplanDecision(
                action="skip",
                reason=f"Hard limit of {self.settings.max_replan_cycles} replan cycles reached",
                failure_dimension="max_cycles",
            )

        eval_result = self.memory.get_evaluation(task.id)
        if not eval_result:
            return ReplanDecision(
                action="retry",
                reason="No evaluation result found — retrying",
                failure_dimension="insufficient_sources",
                revised_question=task.question,
            )

        score = eval_result.score
        suggestions = score.improvement_suggestions

        # Diagnose primary failure dimension
        dims = {
            "factual_grounding": score.factual_grounding,
            "coverage": score.coverage,
            "coherence": score.coherence,
            "citation_quality": score.citation_quality,
        }
        worst_dim = min(dims, key=dims.get)

        if worst_dim in ("factual_grounding", "citation_quality") or score.citation_quality < 4.0:
            return ReplanDecision(
                action="retry",
                reason=f"Low {worst_dim} ({dims[worst_dim]:.1f}): need more/better sources",
                failure_dimension="insufficient_sources",
                revised_question=self._expand_query(task.question),
            )
        elif worst_dim == "coverage" and score.coverage < 5.0:
            return ReplanDecision(
                action="rephrase",
                reason=f"Low coverage ({score.coverage:.1f}): broadening scope",
                failure_dimension="off_topic",
                revised_question=await self._rephrase_for_coverage(task.question, suggestions),
            )
        elif worst_dim == "coherence" and score.coherence < 4.0:
            return ReplanDecision(
                action="retry",
                reason=f"Low coherence ({score.coherence:.1f}): conflicting sources need resolution",
                failure_dimension="contradictory_claims",
                revised_question=task.question,
            )
        else:
            return ReplanDecision(
                action="retry",
                reason=f"Aggregate {score.aggregate:.2f} below threshold — general retry",
                failure_dimension="insufficient_sources",
                revised_question=task.question,
            )

    async def _rephrase_for_coverage(self, question: str, suggestions: list[str]) -> str:
        """Ask the LLM to rephrase the question to improve coverage."""
        loop = asyncio.get_event_loop()
        suggestion_text = "; ".join(suggestions[:2]) if suggestions else "broader scope needed"
        return await loop.run_in_executor(
            None,
            lambda: self._call_llm(
                f"Rephrase this research question to cover missing aspects: {suggestion_text}\n\n"
                f"Original: {question}\n\n"
                "Respond with just the rephrased question, nothing else.",
                max_tokens=2048,
            ),
        )

    def _expand_query(self, question: str) -> str:
        """Simple query expansion — append synonyms/related terms."""
        return f"{question} (include recent developments, technical details, and empirical evidence)"

    async def _merge_task(
        self, task: SubTask, merge_with_id: str
    ) -> Optional[SynthesisResult]:
        """Merge this task's section with a sibling's output."""
        other = self.memory.get_section(merge_with_id)
        if not other:
            return None
        merged_question = f"{task.question} (consolidated with: {other.section_title})"
        task.question = merged_question
        return None  # Will be retried with expanded question

    # ── Query decomposition ──────────────────────────────────────────────────

    async def _decompose_query(self, query: str) -> dict[str, DAGNode]:
        """Decompose the top-level query into a DAG of sub-tasks via LLM."""
        schema = """{
  "tasks": [
    {
      "id": "t1",
      "question": "sub-question text",
      "dependencies": [],
      "complexity": "simple|moderate|complex"
    }
  ],
  "rationale": "brief decomposition rationale"
}"""
        prompt = (
            f"Research query: {query}\n\n"
            "Decompose this into 3-6 focused sub-questions that together fully answer it.\n"
            "Rules:\n"
            "- Use sequential IDs: t1, t2, t3...\n"
            "- Only add a dependency if answering the sub-question REQUIRES a prior answer\n"
            "- Mark complexity: 'simple' for factual lookups, 'moderate' for comparisons, "
            "'complex' for multi-concept synthesis\n"
            "- Aim for parallelism: avoid unnecessary sequential dependencies\n\n"
            f"Respond with JSON:\n{schema}"
        )

        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(
            None,
            lambda: self._call_llm(prompt, max_tokens=4096),
        )

        try:
            raw = raw.strip()
            if raw.startswith("```"):
                lines = raw.split("\n")
                raw = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
            data = json.loads(raw)
            tasks_data = data.get("tasks", [])
            tasks = []
            for td in tasks_data:
                budget_key = _BUDGET_MAP.get(td.get("complexity", "moderate"), "budget_moderate")
                budget = getattr(self.settings, budget_key)
                tasks.append(
                    SubTask(
                        id=td["id"],
                        question=td["question"],
                        dependencies=td.get("dependencies", []),
                        complexity=td.get("complexity", "moderate"),
                        context_budget=budget,
                    )
                )
            dag = build_dag(tasks)
            for t in tasks:
                logger.info("[Orchestrator] Sub-task %s: %s", t.id, t.question)
            logger.info("[Orchestrator] Decomposed into %d tasks", len(dag))
            return dag
        except (json.JSONDecodeError, KeyError) as exc:
            logger.error("[Orchestrator] Decomposition parse failed: %s", exc)
            logger.debug("[Orchestrator] Raw LLM output: %s", raw[:500])
            # Fallback: single task for the whole query
            fallback = SubTask(id="t1", question=query, complexity="complex")
            return build_dag([fallback])

    def _emit_section(self, section: SynthesisResult) -> None:
        """Streaming: notify registered callbacks of a completed section."""
        for cb in self._stream_callbacks:
            try:
                cb(section)
            except Exception:
                pass

    def on_section_complete(self, callback) -> None:
        """Register a streaming callback for section completion."""
        self._stream_callbacks.append(callback)

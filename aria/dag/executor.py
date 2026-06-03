from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Callable, Optional

from aria.schemas.task import DAGNode, SubTask, TaskStatus

logger = logging.getLogger(__name__)


class DAGExecutor:
    """Topological async executor for the research task DAG.

    Executes nodes whose dependencies are all satisfied in parallel
    (asyncio.gather), then continues with newly unblocked nodes.
    Multi-hop support: each node receives the aggregated text output
    of its direct dependencies as `dependency_context`.

    Concurrency is capped at max_parallel to avoid overwhelming APIs.
    """

    def __init__(self, max_parallel: int = 4) -> None:
        self._max_parallel = max_parallel
        self._semaphore: Optional[asyncio.Semaphore] = None

    async def execute(
        self,
        dag: dict[str, DAGNode],
        task_runner: Callable[[SubTask, str], asyncio.Future],
        on_complete: Optional[Callable[[SubTask], None]] = None,
    ) -> dict[str, any]:
        """Execute the DAG and return {task_id: result} for all completed tasks."""
        self._semaphore = asyncio.Semaphore(self._max_parallel)
        results: dict[str, any] = {}
        completed: set[str] = set()
        failed: set[str] = set()

        # Topological execution loop
        max_rounds = len(dag) + 1  # safety: prevents infinite loops on bad graphs
        for _ in range(max_rounds):
            ready = [
                node
                for tid, node in dag.items()
                if tid not in completed
                and tid not in failed
                and node.task.status not in (TaskStatus.COMPLETED, TaskStatus.SKIPPED, TaskStatus.FAILED)
                and all(dep in completed for dep in node.task.dependencies)
            ]

            if not ready:
                if len(completed) + len(failed) < len(dag):
                    self._log_stuck_nodes(dag, completed, failed)
                break

            logger.info(
                "[DAGExecutor] Round: %d ready tasks: %s",
                len(ready),
                [n.task.id for n in ready],
            )

            # Build dependency context for each ready node
            async def run_with_context(node: DAGNode) -> tuple[str, any]:
                dep_context = self._build_dep_context(node, results)
                async with self._semaphore:
                    try:
                        result = await task_runner(node.task, dep_context)
                        return node.task.id, result
                    except Exception as exc:
                        logger.error("[DAGExecutor] Task %s failed: %s", node.task.id, exc, exc_info=True)
                        return node.task.id, None

            batch_results = await asyncio.gather(*[run_with_context(n) for n in ready])

            for task_id, result in batch_results:
                node = dag[task_id]
                if result is not None:
                    results[task_id] = result
                    completed.add(task_id)
                    node.task.status = TaskStatus.COMPLETED
                    if on_complete:
                        on_complete(node.task)
                else:
                    failed.add(task_id)
                    node.task.status = TaskStatus.FAILED

        logger.info(
            "[DAGExecutor] Done: %d completed, %d failed out of %d",
            len(completed), len(failed), len(dag),
        )
        return results

    def _build_dep_context(self, node: DAGNode, results: dict[str, any]) -> str:
        """Build a context string from upstream dependency results."""
        if not node.task.dependencies:
            return ""
        parts = []
        for dep_id in node.task.dependencies:
            dep_result = results.get(dep_id)
            if dep_result and hasattr(dep_result, "content"):
                parts.append(f"[From prerequisite task {dep_id}]:\n{dep_result.content[:600]}")
        return "\n\n".join(parts)

    def _log_stuck_nodes(
        self,
        dag: dict[str, DAGNode],
        completed: set[str],
        failed: set[str],
    ) -> None:
        stuck = [
            tid
            for tid in dag
            if tid not in completed and tid not in failed
        ]
        for tid in stuck:
            node = dag[tid]
            unmet = [d for d in node.task.dependencies if d not in completed]
            logger.warning(
                "[DAGExecutor] Stuck task %s: unmet deps %s (failed deps: %s)",
                tid, unmet, [d for d in unmet if d in failed],
            )


def build_dag(tasks: list[SubTask]) -> dict[str, DAGNode]:
    """Convert a flat list of SubTasks into a DAG adjacency structure."""
    dag: dict[str, DAGNode] = {t.id: DAGNode(task=t) for t in tasks}
    for node in dag.values():
        for dep_id in node.task.dependencies:
            if dep_id in dag:
                dag[dep_id].children.append(node.task.id)
    return dag


def validate_dag(dag: dict[str, DAGNode]) -> list[str]:
    """Return a list of validation errors (empty = valid)."""
    errors = []
    all_ids = set(dag.keys())
    for tid, node in dag.items():
        for dep in node.task.dependencies:
            if dep not in all_ids:
                errors.append(f"Task {tid} depends on unknown task {dep}")

    # Cycle detection via DFS
    visited, rec_stack = set(), set()

    def dfs(tid: str) -> bool:
        visited.add(tid)
        rec_stack.add(tid)
        for child in dag[tid].children:
            if child not in visited:
                if dfs(child):
                    return True
            elif child in rec_stack:
                errors.append(f"Cycle detected involving task {child}")
                return True
        rec_stack.remove(tid)
        return False

    for tid in dag:
        if tid not in visited:
            dfs(tid)

    return errors

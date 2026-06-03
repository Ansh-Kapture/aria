from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

from aria.agents.base import BaseAgent
from aria.config import Settings
from aria.memory.state import SharedWorkingMemory
from aria.schemas.evaluation import EvaluationResult
from aria.schemas.retrieval import Citation
from aria.schemas.synthesis import SynthesisResult

logger = logging.getLogger(__name__)


class ReportGenerator(BaseAgent):
    """Assembles the final structured Markdown research report.

    Produces: executive summary, per-section findings, conflict resolution
    tables, quality scores, and a deduplicated bibliography.
    The structure is deterministic given the same sections — reproducible
    across runs on the same query.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
    ) -> None:
        super().__init__(settings=settings)

    @property
    def system_prompt(self) -> str:
        return (
            "You are a research editor. Given research sections, write a concise "
            "executive summary (200-300 words) that captures the key findings, "
            "highlights areas of uncertainty or conflict, and notes confidence levels. "
            "Be precise and analytical, not promotional."
        )

    async def run(
        self,
        query: str,
        sections: dict[str, SynthesisResult],
        memory: SharedWorkingMemory,
        output_path: Optional[str] = None,
    ) -> str:
        """Generate and optionally write the final Markdown report."""
        evaluations = memory.get_all_evaluations()
        conflicts = memory.get_conflict_resolutions()

        # Sort sections by task ID for reproducible ordering
        ordered_sections = [
            sections[tid]
            for tid in sorted(sections.keys())
            if tid in sections
        ]

        if not ordered_sections:
            return "# Research Report\n\nNo sections were completed.\n"

        # Generate executive summary via LLM
        exec_summary = await self._generate_executive_summary(query, ordered_sections)

        # Build global citation registry (deduplicated)
        global_citations, citation_remap = self._build_citation_registry(ordered_sections)

        # Compute overall confidence
        overall_confidence = self._compute_overall_confidence(evaluations)

        # Assemble report
        parts: list[str] = []
        parts.append(f"# Research Report\n\n**Query:** {query}\n")
        parts.append(f"**Overall Confidence:** {overall_confidence:.1f}/10\n")
        parts.append(f"**Sections Completed:** {len(ordered_sections)}\n")
        parts.append(f"**Conflicts Resolved:** {len(conflicts)}\n\n")
        parts.append("---\n\n")

        # Executive summary
        parts.append("## Executive Summary\n\n")
        parts.append(exec_summary.strip())
        parts.append("\n\n---\n\n")

        # Per-section findings
        parts.append("## Research Findings\n\n")
        for section in ordered_sections:
            eval_result = evaluations.get(section.task_id)
            parts.append(self._render_section(section, eval_result, citation_remap))

        # Conflict resolutions
        if conflicts:
            parts.append("## Conflict Resolutions\n\n")
            parts.append(
                "| Conflict Type | Source A | Source B | Winner | Confidence |\n"
                "|:---|:---|:---|:---|:---|\n"
            )
            for res in conflicts:
                winner_score = (
                    res.score_a.total if res.winner == "a"
                    else res.score_b.total if res.winner == "b"
                    else 0.0
                )
                parts.append(
                    f"| {res.conflict.conflict_type} "
                    f"| {res.conflict.source_a[:40]} "
                    f"| {res.conflict.source_b[:40]} "
                    f"| {res.winner} "
                    f"| {winner_score:.1f}/10 |\n"
                )
            parts.append("\n")

        # Bibliography
        parts.append("## Bibliography\n\n")
        for cit in global_citations:
            parts.append(f"**[{cit.id}]** {cit.title}  \n{cit.url}\n\n")

        # Quality metadata
        parts.append("## Evaluation Metadata\n\n")
        parts.append("```json\n")
        eval_data = {
            tid: {
                "factual_grounding": ev.score.factual_grounding,
                "coverage": ev.score.coverage,
                "coherence": ev.score.coherence,
                "citation_quality": ev.score.citation_quality,
                "aggregate": ev.score.aggregate,
                "verdict": ev.score.verdict,
                "suggestions": ev.score.improvement_suggestions,
            }
            for tid, ev in evaluations.items()
        }
        parts.append(json.dumps(eval_data, indent=2))
        parts.append("\n```\n")

        report = "".join(parts)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(report, encoding="utf-8")
            logger.info("[ReportGenerator] Report written to %s", output_path)

        return report

    def _render_section(
        self,
        section: SynthesisResult,
        eval_result: Optional[EvaluationResult],
        citation_remap: dict[str, str],
    ) -> str:
        parts = [f"### {section.section_title}\n\n"]

        if eval_result:
            s = eval_result.score
            parts.append(
                f"> **Quality:** factual={s.factual_grounding:.1f} | "
                f"coverage={s.coverage:.1f} | "
                f"coherence={s.coherence:.1f} | "
                f"citations={s.citation_quality:.1f} | "
                f"**aggregate={s.aggregate:.2f}**\n\n"
            )

        # Remap citation IDs to global IDs
        content = section.content
        for local_id, global_id in citation_remap.get(section.task_id, {}).items():
            content = content.replace(f"[{local_id}]", f"[{global_id}]")

        parts.append(content)
        parts.append("\n\n")

        # Section-level conflict resolutions
        if section.conflict_resolutions:
            parts.append("**Conflicts resolved in this section:**\n\n")
            for res in section.conflict_resolutions:
                status = "⚠ Unresolved" if res.unresolved else "✓ Resolved"
                parts.append(
                    f"- {status} ({res.conflict.conflict_type}): "
                    f"{res.resolved_claim[:150]}\n"
                )
            parts.append("\n")

        return "".join(parts)

    def _build_citation_registry(
        self, sections: list[SynthesisResult]
    ) -> tuple[list[Citation], dict[str, dict[str, str]]]:
        """Build a global deduplicated citation list and a per-task remap dict."""
        global_by_url: dict[str, Citation] = {}
        counter = 1
        remap: dict[str, dict[str, str]] = {}

        for section in sections:
            remap[section.task_id] = {}
            for cit in section.citations:
                if cit.url not in global_by_url:
                    new_id = str(counter)
                    counter += 1
                    global_cit = cit.model_copy(update={"id": new_id})
                    global_by_url[cit.url] = global_cit
                remap[section.task_id][cit.id] = global_by_url[cit.url].id

        return list(global_by_url.values()), remap

    def _compute_overall_confidence(
        self, evaluations: dict[str, EvaluationResult]
    ) -> float:
        if not evaluations:
            return 0.0
        aggregates = [ev.score.aggregate for ev in evaluations.values()]
        return sum(aggregates) / len(aggregates)

    async def _generate_executive_summary(
        self, query: str, sections: list[SynthesisResult]
    ) -> str:
        sections_text = "\n\n".join(
            f"## {s.section_title}\n{s.content[:400]}" for s in sections[:6]
        )
        prompt = (
            f"Research query: {query}\n\n"
            f"Research sections:\n{sections_text}\n\n"
            "Write a 200-300 word executive summary capturing the key findings, "
            "areas of uncertainty, and overall confidence level."
        )
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._call_llm(prompt, max_tokens=400),
        )

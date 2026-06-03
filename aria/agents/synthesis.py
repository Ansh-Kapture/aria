from __future__ import annotations

import asyncio
import logging
from typing import Optional

from aria.agents.base import BaseAgent
from aria.config import Settings
from aria.schemas.critique import ConflictResolution
from aria.schemas.retrieval import RetrievalResult
from aria.schemas.synthesis import SynthesisResult
from aria.schemas.task import SubTask

logger = logging.getLogger(__name__)


class SynthesisAgent(BaseAgent):
    """Merges retrieval results into coherent, citation-backed prose.

    Deduplicates chunks by chunk_id, assigns sequential citation IDs,
    incorporates conflict resolutions, and produces a structured section.
    Inspired by STORM (Shao et al., 2024) outline+synthesis approach.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        context_budget: Optional[int] = None,
    ) -> None:
        super().__init__(settings=settings, context_budget=context_budget)

    @property
    def system_prompt(self) -> str:
        return (
            "You are an expert research synthesizer. Given retrieved evidence and "
            "conflict resolutions, you write clear, well-structured research sections "
            "with inline citations in [N] format. You deduplicate information, resolve "
            "contradictions per the provided resolutions, and maintain academic tone. "
            "Never fabricate information not present in the provided sources."
        )

    async def run(
        self,
        task: SubTask,
        retrieval_result: RetrievalResult,
        conflict_resolutions: list[ConflictResolution],
        dependency_context: str = "",
    ) -> SynthesisResult:
        """Synthesize retrieval results into a report section."""
        logger.info("[Synthesis] Synthesizing task %s: %s", task.id, task.question)

        # Build citation map: id → Citation
        citation_map = {c.id: c for c in retrieval_result.citations}
        citations_text = "\n".join(
            f"[{c.id}] {c.title} — {c.url}\nSnippet: {c.snippet[:200]}"
            for c in retrieval_result.citations
        )

        # Build evidence text from top chunks
        evidence_parts = []
        seen = set()
        for chunk in retrieval_result.chunks:
            if chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                # Find which citation this chunk belongs to
                matching_cit = next(
                    (c for c in retrieval_result.citations if c.url == chunk.source_url),
                    None,
                )
                cit_ref = f"[{matching_cit.id}]" if matching_cit else ""
                evidence_parts.append(f"Source {cit_ref}: {chunk.text[:400]}")

        evidence_text = "\n\n".join(evidence_parts[:8])

        # Build conflict context
        conflict_text = ""
        if conflict_resolutions:
            conflict_text = "\n\nCONFLICT RESOLUTIONS:\n"
            for r in conflict_resolutions:
                conflict_text += (
                    f"- Conflict ({r.conflict.conflict_type}): "
                    f"Winner={r.winner}. {r.explanation}\n"
                    f"  Resolved claim: {r.resolved_claim}\n"
                )

        # Dependency context from upstream tasks
        dep_context_text = ""
        if dependency_context:
            dep_context_text = f"\n\nCONTEXT FROM PREREQUISITE RESEARCH:\n{dependency_context}\n"

        # Determine section title
        section_title = await self._generate_title(task.question)

        prompt = (
            f"Research question: {task.question}\n\n"
            f"CITATIONS:\n{citations_text}\n\n"
            f"EVIDENCE:\n{evidence_text}"
            f"{conflict_text}"
            f"{dep_context_text}\n\n"
            "Write a comprehensive research section (300-600 words) that:\n"
            "1. Directly answers the research question\n"
            "2. Uses inline citations in [N] format where N is the citation ID\n"
            "3. Incorporates any conflict resolutions as noted above\n"
            "4. Builds on prerequisite context if provided\n"
            "5. Is factually grounded — only state what the evidence supports\n\n"
            "Write the section content only (no title, no header)."
        )

        loop = asyncio.get_event_loop()
        content = await loop.run_in_executor(
            None,
            lambda: self._call_llm(prompt, max_tokens=4096),
        )

        return SynthesisResult(
            task_id=task.id,
            section_title=section_title,
            content=content,
            citations=list(citation_map.values()),
            conflict_resolutions=conflict_resolutions,
        )

    async def _generate_title(self, question: str) -> str:
        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(
            None,
            lambda: self._call_llm(
                f"Convert this research question into a concise section title (5-10 words, title case, no question mark):\n\n{question}",
                max_tokens=256,
            ),
        )
        return raw.strip().strip('"').strip("'")

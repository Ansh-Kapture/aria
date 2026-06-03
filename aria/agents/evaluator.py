from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from aria.agents.base import BaseAgent
from aria.config import Settings
from aria.schemas.evaluation import EvaluationResult, QualityScore
from aria.schemas.synthesis import SynthesisResult
from aria.schemas.task import SubTask

logger = logging.getLogger(__name__)


class QualityEvaluator(BaseAgent):
    """LLM-as-judge quality evaluator for synthesized research sections.

    Scores each section on four dimensions and feeds results back into
    the orchestrator's replanning decisions. Based on LLM-as-Judge
    (Zheng et al., 2023) with a domain-specific rubric.

    Scores are always logged and exposed in the final output.
    """

    def __init__(
        self,
        threshold: Optional[float] = None,
        settings: Optional[Settings] = None,
        context_budget: Optional[int] = None,
    ) -> None:
        super().__init__(settings=settings, context_budget=context_budget)
        self.threshold = threshold or self.settings.quality_threshold

    @property
    def system_prompt(self) -> str:
        return (
            "You are an expert research quality evaluator. Your job is to assess research "
            "sections on four dimensions with numerical scores. Be strict and calibrated: "
            "a 10 means publication-ready quality, 7 means acceptable with minor issues, "
            "5 means significant gaps, below 5 means the section needs to be redone. "
            "Always respond with valid JSON only."
        )

    @property
    def model(self) -> str:
        return self.fast_model

    async def run(
        self, task: SubTask, synthesis: SynthesisResult
    ) -> EvaluationResult:
        """Evaluate a synthesized section and return structured quality scores."""
        logger.info("[Evaluator] Scoring section for task %s", task.id)

        # Truncate content so the prompt stays compact — the evaluator needs
        # enough context to score the section, not the full synthesis text.
        content_preview = synthesis.content[:2000]
        if len(synthesis.content) > 2000:
            content_preview += "\n... [truncated for evaluation]"

        citations_text = "\n".join(
            f"[{c.id}] {c.title}: {c.snippet[:100]}"
            for c in synthesis.citations
        )

        prompt = (
            f"RESEARCH QUESTION: {task.question}\n\n"
            f"SECTION CONTENT:\n{content_preview}\n\n"
            f"AVAILABLE CITATIONS:\n{citations_text or '(none)'}\n\n"
            "Score this research section on:\n"
            "1. factual_grounding (0-10): Are claims backed by the provided citations? "
            "Are there unsupported assertions?\n"
            "2. coverage (0-10): Does the section address all key aspects of the research question? "
            "Are major subtopics missing?\n"
            "3. coherence (0-10): Is the section logically structured with smooth transitions? "
            "Are there internal contradictions?\n"
            "4. citation_quality (0-10): Are citations used appropriately and frequently enough? "
            "Are citation IDs present in the text?\n\n"
            "Also provide:\n"
            "- improvement_suggestions: list of specific, actionable improvements (max 3)\n\n"
            "Respond with JSON:\n"
            '{"factual_grounding": 7.5, "coverage": 6.0, "coherence": 8.0, '
            '"citation_quality": 5.5, "improvement_suggestions": ["...", "..."]}'
        )

        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(
            None,
            lambda: self._call_llm(prompt, max_tokens=2048),
        )

        score = self._parse_score(raw, task.id)
        passed = score.aggregate >= self.threshold

        logger.info(
            "[Evaluator] Task %s: aggregate=%.2f (threshold=%.1f) verdict=%s",
            task.id, score.aggregate, self.threshold, score.verdict,
        )

        return EvaluationResult(
            task_id=task.id,
            section_title=synthesis.section_title,
            score=score,
            passed=passed,
            threshold_used=self.threshold,
        )

    def _parse_score(self, raw: str, task_id: str) -> QualityScore:
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
        try:
            data = json.loads(raw)
            score = QualityScore(
                factual_grounding=float(data.get("factual_grounding", 5.0)),
                coverage=float(data.get("coverage", 5.0)),
                coherence=float(data.get("coherence", 5.0)),
                citation_quality=float(data.get("citation_quality", 5.0)),
                improvement_suggestions=data.get("improvement_suggestions", []),
            )
            # Set verdict based on aggregate
            if score.aggregate >= self.threshold:
                score.verdict = "pass"
            elif score.aggregate < 3.0:
                score.verdict = "skip"
            else:
                score.verdict = "replan"
            return score
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            logger.warning("[Evaluator] Score parse failed for %s: %s", task_id, exc)
            return QualityScore(
                factual_grounding=5.0,
                coverage=5.0,
                coherence=5.0,
                citation_quality=5.0,
                verdict="replan",
                improvement_suggestions=["Re-evaluate: score parsing failed"],
            )

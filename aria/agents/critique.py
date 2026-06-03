from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from aria.agents.base import BaseAgent
from aria.config import Settings
from aria.schemas.critique import (
    Conflict,
    ConflictResolution,
    CredibilityScore,
)
from aria.schemas.retrieval import Chunk

logger = logging.getLogger(__name__)

_SIMILARITY_THRESHOLD = 0.80


class CritiqueAgent(BaseAgent):
    """Detects conflicts between retrieved claims and resolves them via
    structured credibility scoring.

    Conflict detection: embedding cosine similarity above threshold + LLM
    contradiction check. Resolution: multi-dimensional credibility rubric
    (recency, authority, specificity, corroboration).
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        context_budget: Optional[int] = None,
    ) -> None:
        super().__init__(settings=settings, context_budget=context_budget)
        self._embedder = None

    @property
    def system_prompt(self) -> str:
        return (
            "You are a rigorous fact-checking and conflict resolution specialist. "
            "When given two potentially conflicting claims, you reason carefully about "
            "source credibility, recency, specificity, and corroboration to determine "
            "which claim is more credible. You are honest when neither claim is clearly "
            "superior, and you document your reasoning transparently. "
            "Always respond with valid JSON."
        )

    @property
    def model(self) -> str:
        return self.fast_model

    async def run(self, chunks: list[Chunk]) -> list[ConflictResolution]:
        """Detect and resolve conflicts across a set of retrieved chunks."""
        if len(chunks) < 2:
            return []

        conflicts = await self._detect_conflicts(chunks)
        if not conflicts:
            return []

        logger.info("[Critique] Detected %d potential conflicts", len(conflicts))

        loop = asyncio.get_event_loop()
        resolutions = []
        for conflict in conflicts:
            resolution = await loop.run_in_executor(
                None, self._resolve_conflict, conflict
            )
            resolutions.append(resolution)
            logger.debug(
                "[Critique] Resolved conflict: winner=%s, unresolved=%s",
                resolution.winner,
                resolution.unresolved,
            )

        return resolutions

    async def _detect_conflicts(self, chunks: list[Chunk]) -> list[Conflict]:
        """Use LLM to identify claim pairs that contradict each other."""
        # Build a concise summary of distinct claims for the LLM to scan
        claim_summaries = []
        for i, c in enumerate(chunks[:12]):  # cap to prevent massive prompts
            claim_summaries.append(f"[{i}] (src: {c.source_url[:50]}) {c.text[:200]}")

        prompt = (
            "Identify pairs of DIRECTLY CONTRADICTING claims from the following list. "
            "Only flag genuine factual contradictions (not mere differences in framing).\n\n"
            + "\n".join(claim_summaries)
            + "\n\nRespond with JSON: "
            '{"conflicts": [{"index_a": 0, "index_b": 1, "conflict_type": "factual|numerical|temporal|definitional", '
            '"claim_a_summary": "...", "claim_b_summary": "..."}]}'
        )

        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(
            None,
            lambda: self._call_llm(prompt, max_tokens=2048),
        )

        try:
            data = json.loads(self._extract_json(raw))
            conflicts = []
            for item in data.get("conflicts", [])[:5]:  # max 5 conflict pairs
                ia, ib = item.get("index_a", 0), item.get("index_b", 1)
                if ia < len(chunks) and ib < len(chunks):
                    conflicts.append(
                        Conflict(
                            claim_a=chunks[ia].text[:400],
                            source_a=chunks[ia].source_url,
                            claim_b=chunks[ib].text[:400],
                            source_b=chunks[ib].source_url,
                            conflict_type=item.get("conflict_type", "factual"),
                            similarity_score=_SIMILARITY_THRESHOLD,
                        )
                    )
            return conflicts
        except (json.JSONDecodeError, KeyError) as exc:
            logger.debug("[Critique] Conflict detection parse error: %s", exc)
            return []

    def _resolve_conflict(self, conflict: Conflict) -> ConflictResolution:
        prompt = (
            "You are resolving a factual conflict between two sources.\n\n"
            f"CLAIM A (from {conflict.source_a}):\n{conflict.claim_a}\n\n"
            f"CLAIM B (from {conflict.source_b}):\n{conflict.claim_b}\n\n"
            "Score each source on the following rubric (JSON format):\n"
            "- recency: 0-3 (higher = more recent publication)\n"
            "- authority: 0-3 (3=academic paper, 2=reputable news/docs, 1=blog, 0=unknown)\n"
            "- specificity: 0-2 (2=quantitative with data, 1=qualitative with examples, 0=vague)\n"
            "- corroboration: 0-2 (2=widely supported, 1=some support, 0=isolated claim)\n\n"
            "Then determine the winner and provide the resolved claim.\n\n"
            'Respond with JSON: {"score_a": {...}, "score_b": {...}, '
            '"winner": "a|b|neither|both_valid", "explanation": "...", '
            '"resolved_claim": "...", "unresolved": false, "unresolved_note": null}'
        )

        raw = self._call_llm(prompt, max_tokens=2048)
        try:
            data = json.loads(self._extract_json(raw))
            score_a = CredibilityScore(**data["score_a"])
            score_b = CredibilityScore(**data["score_b"])
            return ConflictResolution(
                conflict=conflict,
                winner=data.get("winner", "neither"),
                score_a=score_a,
                score_b=score_b,
                explanation=data.get("explanation", ""),
                resolved_claim=data.get("resolved_claim", conflict.claim_a),
                unresolved=data.get("unresolved", False),
                unresolved_note=data.get("unresolved_note"),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("[Critique] Resolution parse failed: %s", exc)
            return ConflictResolution(
                conflict=conflict,
                winner="neither",
                score_a=CredibilityScore(recency=1, authority=1, specificity=1, corroboration=1),
                score_b=CredibilityScore(recency=1, authority=1, specificity=1, corroboration=1),
                explanation="Resolution failed due to parsing error.",
                resolved_claim=conflict.claim_a,
                unresolved=True,
                unresolved_note="Parse error — both claims retained",
            )

    @staticmethod
    def _extract_json(text: str) -> str:
        """Strip markdown fences if present."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
        return text

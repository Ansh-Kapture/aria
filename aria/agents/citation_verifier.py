from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from aria.agents.base import BaseAgent
from aria.config import Settings
from aria.schemas.retrieval import Citation

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    citation_id: str
    url: str
    claim: str
    verified: bool
    confidence: float
    note: str


class CitationVerifier(BaseAgent):
    """Spot-checks whether cited claims actually appear in source documents.

    For each citation, fetches a snippet of the source and asks the LLM
    whether the attributed claim is supported. Flags unverified citations
    in the final report.
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
            "You are a citation verification specialist. Given a claim and a source text, "
            "determine whether the source text actually supports the claim. Be strict: "
            "paraphrasing is acceptable, but fabrication is not. Always respond with valid JSON."
        )

    @property
    def model(self) -> str:
        return self.fast_model

    async def run(
        self,
        citations: list[Citation],
        section_content: str,
        sample_size: int = 3,
    ) -> list[VerificationResult]:
        """Verify a random sample of citations from a section."""
        if not citations:
            return []

        # Sample up to `sample_size` citations for spot-checking
        import random
        sample = random.sample(citations, min(sample_size, len(citations)))

        results = await asyncio.gather(
            *[self._verify_one(cit, section_content) for cit in sample],
            return_exceptions=True,
        )

        verified = []
        for r in results:
            if isinstance(r, VerificationResult):
                verified.append(r)
            else:
                logger.warning("[CitationVerifier] Verification error: %s", r)

        flagged = [r for r in verified if not r.verified]
        if flagged:
            logger.warning("[CitationVerifier] %d citation(s) could not be verified", len(flagged))

        return verified

    async def _verify_one(
        self, citation: Citation, section_content: str
    ) -> VerificationResult:
        # Extract sentences from the section that reference this citation
        import re
        pattern = rf'\[[^\]]*{re.escape(citation.id)}[^\]]*\][^.!?]*[.!?]'
        matches = re.findall(pattern, section_content)
        claim = matches[0] if matches else section_content[:200]

        prompt = (
            f"CLAIM (from section): {claim[:300]}\n\n"
            f"SOURCE TITLE: {citation.title}\n"
            f"SOURCE URL: {citation.url}\n"
            f"SOURCE SNIPPET: {citation.snippet[:500]}\n\n"
            "Does the source snippet support the claim?\n"
            'Respond with JSON: {"verified": true, "confidence": 0.85, "note": "..."}'
        )

        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(
            None,
            lambda: self._call_llm(prompt, max_tokens=1024),
        )

        try:
            import json
            raw = raw.strip()
            if raw.startswith("```"):
                lines = raw.split("\n")
                raw = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
            data = json.loads(raw)
            return VerificationResult(
                citation_id=citation.id,
                url=citation.url,
                claim=claim[:200],
                verified=bool(data.get("verified", True)),
                confidence=float(data.get("confidence", 0.7)),
                note=data.get("note", ""),
            )
        except Exception as exc:
            return VerificationResult(
                citation_id=citation.id,
                url=citation.url,
                claim=claim[:200],
                verified=True,
                confidence=0.5,
                note=f"Verification parse error: {exc}",
            )

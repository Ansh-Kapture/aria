"""Tests for CritiqueAgent conflict detection and resolution."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from aria.agents.critique import CritiqueAgent
from aria.memory.state import make_chunk_id
from aria.schemas.critique import (
    Conflict,
    ConflictResolution,
    CredibilityScore,
)
from aria.schemas.retrieval import Chunk


def make_chunk(text: str, url: str) -> Chunk:
    return Chunk(
        text=text,
        source_url=url,
        source_title="Test",
        relevance_score=0.8,
        chunk_id=make_chunk_id(text, url),
    )


@pytest.fixture
def agent():
    from aria.config import Settings
    settings = Settings(anthropic_api_key="test-key")
    return CritiqueAgent(settings=settings)


@pytest.fixture
def mock_llm(agent):
    """Patch _call_llm to return predictable JSON."""
    def fake_llm(prompt, **kwargs):
        if "contradicting" in prompt.lower() or "conflicts" in prompt.lower():
            return json.dumps({
                "conflicts": [
                    {
                        "index_a": 0,
                        "index_b": 1,
                        "conflict_type": "factual",
                        "claim_a_summary": "A says X",
                        "claim_b_summary": "B says not-X",
                    }
                ]
            })
        elif "resolving" in prompt.lower() or "score_a" in prompt.lower() or "winner" in prompt.lower():
            return json.dumps({
                "score_a": {"recency": 2.0, "authority": 2.0, "specificity": 1.5, "corroboration": 1.0},
                "score_b": {"recency": 1.0, "authority": 1.0, "specificity": 1.0, "corroboration": 0.5},
                "winner": "a",
                "explanation": "Source A is more authoritative and recent.",
                "resolved_claim": "Transformers have quadratic attention complexity.",
                "unresolved": False,
                "unresolved_note": None,
            })
        return "{}"

    with patch.object(agent, "_call_llm", side_effect=fake_llm):
        yield agent


@pytest.mark.asyncio
async def test_detect_conflicts(mock_llm):
    chunks = [
        make_chunk("Transformers have O(n²) attention complexity.", "http://paper1.com"),
        make_chunk("Transformers have linear attention complexity with approximations.", "http://paper2.com"),
    ]
    resolutions = await mock_llm.run(chunks)
    assert len(resolutions) == 1
    assert resolutions[0].winner == "a"
    assert resolutions[0].conflict.conflict_type == "factual"


@pytest.mark.asyncio
async def test_no_conflicts_with_single_chunk(mock_llm):
    chunks = [make_chunk("Single claim.", "http://only.com")]
    resolutions = await mock_llm.run(chunks)
    assert resolutions == []


@pytest.mark.asyncio
async def test_credibility_scoring(mock_llm):
    chunks = [
        make_chunk("Claim A from reputable source.", "http://a.com"),
        make_chunk("Claim B from blog post.", "http://b.com"),
    ]
    resolutions = await mock_llm.run(chunks)
    assert len(resolutions) == 1
    res = resolutions[0]
    # Source A should score higher
    assert res.score_a.total > res.score_b.total


@pytest.mark.asyncio
async def test_unresolved_conflict_handling():
    """When neither source is clearly credible, resolution is marked unresolved."""
    from aria.config import Settings
    settings = Settings(anthropic_api_key="test-key")
    agent = CritiqueAgent(settings=settings)

    unresolved_response = json.dumps({
        "score_a": {"recency": 1.0, "authority": 1.0, "specificity": 1.0, "corroboration": 1.0},
        "score_b": {"recency": 1.0, "authority": 1.0, "specificity": 1.0, "corroboration": 1.0},
        "winner": "neither",
        "explanation": "Both sources have similar credibility.",
        "resolved_claim": "The evidence is conflicting.",
        "unresolved": True,
        "unresolved_note": "Both claims retained",
    })

    conflict = Conflict(
        claim_a="Claim A",
        source_a="http://a.com",
        claim_b="Claim B",
        source_b="http://b.com",
        conflict_type="factual",
        similarity_score=0.85,
    )

    with patch.object(agent, "_call_llm", return_value=unresolved_response):
        resolution = agent._resolve_conflict(conflict)

    assert resolution.unresolved is True
    assert resolution.winner == "neither"
    assert resolution.unresolved_note is not None


def test_credibility_score_total():
    score = CredibilityScore(recency=2.5, authority=3.0, specificity=2.0, corroboration=1.5)
    assert score.total == pytest.approx(9.0)


def test_extract_json_strips_markdown():
    raw = "```json\n{\"key\": \"value\"}\n```"
    result = CritiqueAgent._extract_json(raw)
    assert result.strip() == '{"key": "value"}'


def test_resolve_conflict_parse_failure():
    """CritiqueAgent should return a safe fallback on parse error."""
    from aria.config import Settings
    settings = Settings(anthropic_api_key="test-key")
    agent = CritiqueAgent(settings=settings)
    conflict = Conflict(
        claim_a="A", source_a="http://a.com",
        claim_b="B", source_b="http://b.com",
        conflict_type="factual", similarity_score=0.9,
    )
    with patch.object(agent, "_call_llm", return_value="not valid json at all"):
        resolution = agent._resolve_conflict(conflict)
    assert resolution.unresolved is True
    assert resolution.winner == "neither"

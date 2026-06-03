from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Provider-agnostic LLM interface used by all agents.

    Each concrete implementation wraps one SDK (Anthropic, OpenAI,
    Vertex AI, or an OpenAI-compatible endpoint) and exposes a single
    synchronous `complete()` method. Agents call this via an executor
    thread to remain async-safe.
    """

    @abstractmethod
    def complete(
        self,
        system: str,
        user: str,
        model: str,
        max_tokens: int,
        temperature: float = 0.3,
    ) -> str:
        """Return the assistant's text response."""

    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider identifier for logging."""

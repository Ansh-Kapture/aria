from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from aria.config import Settings, get_settings
from aria.providers.base import LLMProvider
from aria.providers.factory import create_provider

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base for all ARIA agents.

    Provider-agnostic: uses a `LLMProvider` that abstracts over Anthropic,
    OpenAI, OpenAI-compatible endpoints, and Vertex AI. All agents call
    `_call_llm()` / `_call_llm_structured()` — no SDK-specific code here.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        context_budget: Optional[int] = None,
        provider: Optional[LLMProvider] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.context_budget = context_budget or self.settings.budget_moderate
        self._provider: Optional[LLMProvider] = provider

    @property
    def provider(self) -> LLMProvider:
        if self._provider is None:
            self._provider = create_provider(self.settings)
        return self._provider

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """The agent's specialised system prompt."""

    @property
    def model(self) -> str:
        return self.settings.get_orchestrator_model()

    @property
    def fast_model(self) -> str:
        return self.settings.get_fast_model()

    def _call_llm(
        self,
        user_message: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.3,
    ) -> str:
        """Synchronous LLM call — used as executor target in async contexts."""
        used_model = model or self.model
        used_max_tokens = min(max_tokens or self.context_budget, self.context_budget)

        logger.debug(
            "[%s] %s model=%s budget=%d",
            self.__class__.__name__,
            self.provider.provider_name(),
            used_model,
            used_max_tokens,
        )

        return self.provider.complete(
            system=self.system_prompt,
            user=user_message,
            model=used_model,
            max_tokens=used_max_tokens,
            temperature=temperature,
        )

    def _call_llm_structured(
        self,
        user_message: str,
        schema_description: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """LLM call that instructs the model to respond with valid JSON."""
        json_instruction = (
            "\n\nRespond with valid JSON only (no markdown fences, no explanation) "
            f"matching this schema:\n{schema_description}"
        )
        return self._call_llm(
            user_message + json_instruction,
            model=model,
            max_tokens=max_tokens,
            temperature=0.1,
        )

    @abstractmethod
    async def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the agent's primary task."""

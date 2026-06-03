from __future__ import annotations

import logging

from aria.providers.base import LLMProvider

logger = logging.getLogger(__name__)


def create_provider(settings) -> LLMProvider:
    """Instantiate the correct LLMProvider based on settings.

    Provider selection order (first configured wins):
      1. vertexai  — if VERTEX_PROJECT is set
      2. openai    — if OPENAI_API_KEY is set (or OPENAI_BASE_URL for custom endpoints)
      3. anthropic — if ANTHROPIC_API_KEY is set  (default)

    Override explicitly with LLM_PROVIDER=anthropic|openai|openai_compatible|vertexai
    """
    provider_name = settings.llm_provider.lower()

    if provider_name == "auto":
        provider_name = _auto_detect(settings)

    logger.info("[LLMProvider] Using provider: %s", provider_name)

    if provider_name == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is required for the 'anthropic' provider. "
                "Set it in .env or choose a different provider."
            )
        from aria.providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider(api_key=settings.anthropic_api_key)

    if provider_name in ("openai", "openai_compatible"):
        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required for the 'openai' / 'openai_compatible' provider."
            )
        from aria.providers.openai_provider import OpenAIProvider
        return OpenAIProvider(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url or None,
        )

    if provider_name == "vertexai":
        if not settings.vertex_project:
            raise ValueError(
                "VERTEX_PROJECT is required for the 'vertexai' provider."
            )
        from aria.providers.vertexai_provider import VertexAIProvider
        return VertexAIProvider(
            project=settings.vertex_project,
            location=settings.vertex_location,
            service_account_json=settings.vertex_service_account_json,
            service_account_file=settings.vertex_service_account_file,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER: {provider_name!r}. "
        "Valid values: anthropic, openai, openai_compatible, vertexai, auto"
    )


def _auto_detect(settings) -> str:
    """Pick a provider automatically from whichever credentials are present."""
    if settings.vertex_project:
        return "vertexai"
    if settings.openai_api_key:
        return "openai"
    if settings.anthropic_api_key:
        return "anthropic"
    raise ValueError(
        "No LLM credentials found. Set one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, "
        "or VERTEX_PROJECT (+ optional VERTEX_SERVICE_ACCOUNT_JSON). "
        "See .env.example for details."
    )

from __future__ import annotations

from typing import Optional

from aria.providers.base import LLMProvider


class OpenAIProvider(LLMProvider):
    """OpenAI API provider.

    Also works for any OpenAI-compatible endpoint (Ollama, vLLM, LM Studio,
    Together AI, Groq, etc.) by setting `base_url`.

    Examples:
        # OpenAI
        OpenAIProvider(api_key="sk-...")

        # Ollama local
        OpenAIProvider(api_key="ollama", base_url="http://localhost:11434/v1")

        # vLLM
        OpenAIProvider(api_key="EMPTY", base_url="http://localhost:8000/v1")

        # Together AI
        OpenAIProvider(api_key="...", base_url="https://api.together.xyz/v1")
    """

    def __init__(self, api_key: str, base_url: Optional[str] = None) -> None:
        from openai import OpenAI
        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self._base_url = base_url

    def complete(
        self,
        system: str,
        user: str,
        model: str,
        max_tokens: int,
        temperature: float = 0.3,
    ) -> str:
        response = self._client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""

    def provider_name(self) -> str:
        if self._base_url:
            return f"openai-compatible ({self._base_url})"
        return "openai"

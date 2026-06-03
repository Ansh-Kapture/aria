from __future__ import annotations

import json
import logging
import os

from aria.providers.base import LLMProvider

logger = logging.getLogger(__name__)

# Gemini 2.x thinking models burn reasoning tokens inside max_output_tokens.
# Raise small budgets so there is always room for actual response text.
_THINKING_MODEL_MIN_TOKENS = 8192
_THINKING_MODEL_MARKERS = ("gemini-2.", "thinking")


def _is_thinking_model(model: str) -> bool:
    return any(m in model.lower() for m in _THINKING_MODEL_MARKERS)


class VertexAIProvider(LLMProvider):
    """Google Vertex AI provider via the vertexai SDK.

    Credentials (tried in order):
    1. service_account_json — raw JSON string
    2. service_account_file — path to a .json file on disk
    3. Application Default Credentials — works inside GCP or after
       `gcloud auth application-default login`

    Thinking-model handling:
        Gemini 2.x models spend tokens on internal reasoning before writing
        output; those tokens count against max_output_tokens. This provider
        raises any budget below _THINKING_MODEL_MIN_TOKENS to ensure agents
        always receive a usable response.
    """

    def __init__(
        self,
        project: str,
        location: str = "us-central1",
        service_account_json: str = "",
        service_account_file: str = "",
    ) -> None:
        self._project = project
        self._location = location
        self._sa_json = service_account_json
        self._sa_file = service_account_file
        self._initialized = False

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return

        import vertexai
        from google.oauth2 import service_account

        credentials = None

        if self._sa_json:
            sa_data = json.loads(self._sa_json)
            credentials = service_account.Credentials.from_service_account_info(
                sa_data,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
        elif self._sa_file and os.path.isfile(self._sa_file):
            credentials = service_account.Credentials.from_service_account_file(
                self._sa_file,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )

        vertexai.init(
            project=self._project,
            location=self._location,
            credentials=credentials,
        )
        self._initialized = True

    def complete(
        self,
        system: str,
        user: str,
        model: str,
        max_tokens: int,
        temperature: float = 0.3,
    ) -> str:
        self._ensure_initialized()
        from vertexai.generative_models import GenerativeModel, GenerationConfig

        effective_max_tokens = max_tokens
        if _is_thinking_model(model) and max_tokens < _THINKING_MODEL_MIN_TOKENS:
            logger.debug(
                "Raising max_tokens %d → %d for thinking model %s",
                max_tokens, _THINKING_MODEL_MIN_TOKENS, model,
            )
            effective_max_tokens = _THINKING_MODEL_MIN_TOKENS

        generation_config = GenerationConfig(
            max_output_tokens=effective_max_tokens,
            temperature=temperature,
        )

        llm = GenerativeModel(
            model_name=model,
            system_instruction=system,
            generation_config=generation_config,
        )

        response = llm.generate_content(user)

        try:
            return response.text
        except ValueError:
            try:
                parts = response.candidates[0].content.parts
                if parts:
                    partial = "".join(p.text for p in parts if hasattr(p, "text"))
                    if partial.strip():
                        logger.warning(
                            "Model %s hit MAX_TOKENS — partial response (%d chars)",
                            model, len(partial),
                        )
                        return partial
            except Exception:
                pass
            finish = "unknown"
            try:
                finish = response.candidates[0].finish_reason.name
            except Exception:
                pass
            raise RuntimeError(
                f"Vertex AI model {model!r} returned no text "
                f"(finish_reason={finish}). "
                "Switch to a non-thinking variant (e.g. gemini-1.5-pro) "
                "or increase token limits."
            ) from None

    def provider_name(self) -> str:
        return f"vertexai (project={self._project}, location={self._location})"

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # ── Provider selection ───────────────────────────────────────────────────
    # "auto" = pick from whichever key is present (vertexai > openai > anthropic)
    llm_provider: str = "auto"

    # ── Anthropic ────────────────────────────────────────────────────────────
    anthropic_api_key: str = ""
    # Main (orchestrator/synthesis/retrieval) model
    orchestrator_model: str = "claude-sonnet-4-6"
    # Lightweight (evaluator/critique) model
    fast_model: str = "claude-haiku-4-5-20251001"

    # ── OpenAI / OpenAI-compatible endpoint ──────────────────────────────────
    openai_api_key: str = ""
    # Leave blank for official OpenAI. For Ollama: http://localhost:11434/v1
    # For vLLM: http://localhost:8000/v1  For Together: https://api.together.xyz/v1
    openai_base_url: str = ""
    openai_model: str = "gpt-4o"
    openai_fast_model: str = "gpt-4o-mini"

    # ── Vertex AI ─────────────────────────────────────────────────────────────
    vertex_project: str = ""
    vertex_location: str = "us-central1"
    vertex_model: str = "gemini-1.5-pro"
    vertex_fast_model: str = "gemini-1.5-flash"
    # Paste the full service-account JSON as a single-line string, OR point to a file
    vertex_service_account_json: str = ""   # raw JSON string
    vertex_service_account_file: str = ""   # path to .json file on disk

    # ── Web search ───────────────────────────────────────────────────────────
    tavily_api_key: str = ""
    use_tavily: bool = False

    # ── Vector store ─────────────────────────────────────────────────────────
    chroma_persist_dir: str = "./state/chroma"
    collection_name: str = "aria_research"

    # ── Embeddings (local sentence-transformers — no API key needed) ─────────
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── Orchestration ────────────────────────────────────────────────────────
    max_replan_cycles: int = 3
    quality_threshold: float = 6.5
    global_timeout_seconds: int = 600
    max_parallel_agents: int = 4

    # ── Context budgets by complexity ────────────────────────────────────────
    budget_simple: int = 2048
    budget_moderate: int = 6144
    budget_complex: int = 12288

    # ── Retrieval ────────────────────────────────────────────────────────────
    top_k_retrieval: int = 8
    top_k_rerank: int = 4
    hyde_iterations: int = 2
    min_confidence_for_retry: float = 0.55

    # ── Output ───────────────────────────────────────────────────────────────
    output_dir: str = "./output"
    state_dir: str = "./state"

    def get_orchestrator_model(self) -> str:
        """Return the correct main model name for the active provider."""
        p = self.llm_provider.lower()
        if p in ("openai", "openai_compatible"):
            return self.openai_model
        if p == "vertexai":
            return self.vertex_model
        return self.orchestrator_model  # anthropic default

    def get_fast_model(self) -> str:
        """Return the correct lightweight model name for the active provider."""
        p = self.llm_provider.lower()
        if p in ("openai", "openai_compatible"):
            return self.openai_fast_model
        if p == "vertexai":
            return self.vertex_fast_model
        return self.fast_model  # anthropic default


@lru_cache
def get_settings() -> Settings:
    return Settings()

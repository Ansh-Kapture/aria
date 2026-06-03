from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
from pathlib import Path
from typing import Callable, Optional

from aria.schemas.retrieval import Chunk

logger = logging.getLogger(__name__)


class VectorStoreTool:
    """ChromaDB-backed semantic store with local sentence-transformer embeddings.

    Supports HyDE: the caller provides a hypothesis string which is embedded
    instead of the raw question for better semantic match to real documents.
    Also provides cross-encoder reranking to refine top-k candidates.
    """

    def __init__(
        self,
        persist_dir: str = "./state/chroma",
        collection_name: str = "aria_research",
        embedding_model: str = "all-MiniLM-L6-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._embedding_model_name = embedding_model
        self._reranker_model_name = reranker_model
        self._collection = None
        self._embedder = None
        self._reranker = None
        self._initialized = False
        self._init_lock = threading.Lock()  # guards concurrent thread-pool init

    def _ensure_initialized(self) -> None:
        # Fast path — already up
        if self._initialized:
            return
        # Slow path — acquire lock so only one thread runs ChromaDB init
        with self._init_lock:
            if self._initialized:  # double-check after acquiring
                return

            import chromadb
            from sentence_transformers import SentenceTransformer, CrossEncoder

            Path(self._persist_dir).mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=self._persist_dir)
        self._collection = client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder = SentenceTransformer(self._embedding_model_name)
        try:
            self._reranker = CrossEncoder(self._reranker_model_name)
        except Exception as exc:
            logger.warning("Reranker not available (%s), skipping rerank step", exc)
            self._reranker = None

        self._initialized = True
        logger.info(
            "VectorStore initialized: collection=%s, docs=%d",
            self._collection_name,
            self._collection.count(),
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        self._ensure_initialized()
        return self._embedder.encode(texts, normalize_embeddings=True).tolist()

    # ── Store ────────────────────────────────────────────────────────────────

    async def store(self, chunks: list[Chunk]) -> None:
        """Embed and persist chunks into ChromaDB."""
        if not chunks:
            return
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._sync_store, chunks)

    def _sync_store(self, chunks: list[Chunk]) -> None:
        self._ensure_initialized()
        texts = [c.text for c in chunks]
        embeddings = self._embed(texts)
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {
                    "source_url": c.source_url,
                    "source_title": c.source_title,
                    "relevance_score": c.relevance_score,
                    "position": c.position,
                }
                for c in chunks
            ],
        )

    # ── Query ────────────────────────────────────────────────────────────────

    async def query(self, text: str, top_k: int = 8) -> list[Chunk]:
        """Retrieve top-k semantically similar chunks."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_query, text, top_k)

    def _sync_query(self, text: str, top_k: int) -> list[Chunk]:
        self._ensure_initialized()
        if self._collection.count() == 0:
            return []

        embeddings = self._embed([text])
        results = self._collection.query(
            query_embeddings=embeddings,
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            similarity = 1.0 - dist  # cosine distance → similarity
            chunks.append(
                Chunk(
                    text=doc,
                    source_url=meta.get("source_url", ""),
                    source_title=meta.get("source_title", ""),
                    relevance_score=max(0.0, min(1.0, similarity)),
                    chunk_id=hashlib.sha256(doc[:200].encode()).hexdigest()[:16],
                    position=meta.get("position", 0),
                )
            )
        return chunks

    # ── HyDE query ───────────────────────────────────────────────────────────

    async def query_with_hyde(
        self,
        question: str,
        llm_fn: Callable[[str], str],
        top_k: int = 8,
    ) -> tuple[list[Chunk], str]:
        """HyDE: embed a hypothetical answer instead of the raw question.

        Returns (chunks, hypothesis_used).
        """
        hypothesis_prompt = (
            f"Write a concise, factually plausible answer to this question "
            f"(2-3 sentences, include specific technical details):\n\n{question}"
        )
        loop = asyncio.get_event_loop()
        hypothesis = await loop.run_in_executor(None, llm_fn, hypothesis_prompt)
        chunks = await self.query(hypothesis, top_k=top_k)
        return chunks, hypothesis

    # ── Rerank ───────────────────────────────────────────────────────────────

    async def rerank(self, query: str, chunks: list[Chunk], top_k: int = 4) -> list[Chunk]:
        """Re-score chunks with a cross-encoder and return top-k."""
        if not chunks or self._reranker is None:
            return chunks[:top_k]
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_rerank, query, chunks, top_k)

    def _sync_rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]:
        self._ensure_initialized()
        if self._reranker is None:
            return sorted(chunks, key=lambda c: c.relevance_score, reverse=True)[:top_k]

        pairs = [(query, c.text) for c in chunks]
        scores = self._reranker.predict(pairs)
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        return [
            # Cross-encoder scores are logits (unbounded); clamp to [0, 1]
            c.model_copy(update={"relevance_score": max(0.0, min(1.0, float(score)))})
            for score, c in ranked[:top_k]
        ]

    def count(self) -> int:
        self._ensure_initialized()
        return self._collection.count()

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from aria.agents.base import BaseAgent
from aria.config import Settings
from aria.memory.state import SharedWorkingMemory, make_chunk_id
from aria.schemas.retrieval import Chunk, Citation, RetrievalResult
from aria.schemas.task import SubTask
from aria.tools.vector_store import VectorStoreTool
from aria.tools.web_search import WebSearchTool

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 600   # ~approx words per chunk
_CHUNK_OVERLAP = 80


def _chunk_text(text: str, url: str, title: str) -> list[Chunk]:
    words = text.split()
    chunks = []
    for i in range(0, max(1, len(words) - _CHUNK_OVERLAP), _CHUNK_SIZE - _CHUNK_OVERLAP):
        window = words[i : i + _CHUNK_SIZE]
        if len(window) < 50:
            continue
        chunk_text = " ".join(window)
        chunks.append(
            Chunk(
                text=chunk_text,
                source_url=url,
                source_title=title,
                relevance_score=0.0,
                chunk_id=make_chunk_id(chunk_text, url),
                position=i,
            )
        )
    return chunks


class RetrievalAgent(BaseAgent):
    """Fetches, chunks, and ranks content for a single sub-task.

    Strategy: HyDE → embed hypothesis → retrieve from vector store + web →
    cross-encoder rerank → iterative follow-up if confidence is low.
    Inspired by Self-RAG (Asai et al., 2023) and HyDE (Gao et al., 2022).
    """

    def __init__(
        self,
        web_search: Optional[WebSearchTool] = None,
        vector_store: Optional[VectorStoreTool] = None,
        settings: Optional[Settings] = None,
        context_budget: Optional[int] = None,
    ) -> None:
        super().__init__(settings=settings, context_budget=context_budget)
        self._web = web_search or WebSearchTool(
            tavily_api_key=self.settings.tavily_api_key,
            use_tavily=self.settings.use_tavily,
        )
        self._vs = vector_store or VectorStoreTool(
            persist_dir=self.settings.chroma_persist_dir,
            collection_name=self.settings.collection_name,
            embedding_model=self.settings.embedding_model,
            reranker_model=self.settings.reranker_model,
        )

    @property
    def system_prompt(self) -> str:
        return (
            "You are a precise research retrieval specialist. "
            "Your role is to generate concise, factually plausible hypothetical answers "
            "to research questions. These hypotheses improve semantic search quality. "
            "Be specific, technical, and include concrete details when possible."
        )

    async def run(
        self, task: SubTask, memory: SharedWorkingMemory
    ) -> RetrievalResult:
        logger.info("[Retrieval] Starting task %s: %s", task.id, task.question)

        # Step 1: Generate HyDE hypothesis
        loop = asyncio.get_event_loop()
        hypothesis = await loop.run_in_executor(
            None,
            lambda: self._call_llm(
                f"Write a concise, factually plausible 2-3 sentence answer to:\n\n{task.question}",
                max_tokens=2048,
            ),
        )
        logger.debug("[Retrieval] HyDE hypothesis: %s", hypothesis[:100])

        # Step 2: Retrieve from vector store + web search in parallel
        vs_chunks, web_results = await asyncio.gather(
            self._vs.query(hypothesis, top_k=self.settings.top_k_retrieval),
            self._web.search(task.question),
        )

        # Step 3: Chunk web results and store in vector store
        new_web_chunks: list[Chunk] = []
        for result in web_results:
            text = result.full_text or result.snippet
            if not text:
                continue
            page_chunks = _chunk_text(text, result.url, result.title)
            new_web_chunks.extend(page_chunks)

        await self._vs.store(new_web_chunks)

        # Step 4: Re-query with HyDE hypothesis to include newly stored chunks
        fresh_chunks = await self._vs.query(hypothesis, top_k=self.settings.top_k_retrieval)

        # Step 5: Cross-encoder reranking
        all_candidates = {c.chunk_id: c for c in fresh_chunks + vs_chunks}.values()
        reranked = await self._vs.rerank(
            task.question,
            list(all_candidates),
            top_k=self.settings.top_k_rerank,
        )

        # Step 6: Compute confidence
        confidence = self._compute_confidence(reranked)

        # Step 7: Iterative follow-up if confidence is low
        iterations = 1
        if (
            confidence < self.settings.min_confidence_for_retry
            and iterations < self.settings.hyde_iterations
        ):
            logger.info("[Retrieval] Low confidence (%.2f), running follow-up retrieval", confidence)
            follow_up_query = await loop.run_in_executor(
                None,
                lambda: self._call_llm(
                    f"Extract 3-5 key technical terms from these snippets for follow-up search:\n"
                    + "\n".join(c.text[:100] for c in reranked[:3])
                    + f"\n\nOriginal question: {task.question}",
                    max_tokens=512,
                ),
            )
            follow_up_results = await self._web.search(follow_up_query)
            follow_up_chunks = []
            for r in follow_up_results[:3]:
                follow_up_chunks.extend(_chunk_text(r.full_text or r.snippet, r.url, r.title))
            await self._vs.store(follow_up_chunks)
            fresh2 = await self._vs.query(hypothesis, top_k=self.settings.top_k_retrieval)
            reranked = await self._vs.rerank(task.question, fresh2, top_k=self.settings.top_k_rerank)
            confidence = self._compute_confidence(reranked)
            iterations = 2

        # Step 8: Store new chunks in working memory (dedup handled internally)
        all_retrieved = reranked + new_web_chunks
        await memory.store_chunks(all_retrieved)

        # Step 9: Build citations from unique sources
        citations = self._build_citations(reranked, web_results)

        logger.info(
            "[Retrieval] Task %s: %d chunks, %d citations, confidence=%.2f",
            task.id, len(reranked), len(citations), confidence,
        )

        return RetrievalResult(
            task_id=task.id,
            chunks=reranked,
            citations=citations,
            confidence=confidence,
            hyde_hypothesis=hypothesis,
            iterations=iterations,
        )

    def _compute_confidence(self, chunks: list[Chunk]) -> float:
        if not chunks:
            return 0.0
        scores = [c.relevance_score for c in chunks]
        raw = sum(scores) / len(scores)
        # Cross-encoder scores are unbounded; clamp to [0, 1] for the schema
        return max(0.0, min(1.0, raw))

    def _build_citations(self, chunks: list[Chunk], web_results) -> list[Citation]:
        seen_urls: set[str] = set()
        url_to_snippet: dict[str, str] = {r.url: r.snippet for r in web_results}
        citations = []
        for chunk in chunks:
            if chunk.source_url and chunk.source_url not in seen_urls:
                seen_urls.add(chunk.source_url)
                cit_id = f"c{len(citations) + 1}"
                citations.append(
                    Citation(
                        id=cit_id,
                        url=chunk.source_url,
                        title=chunk.source_title,
                        snippet=url_to_snippet.get(chunk.source_url, chunk.text[:200]),
                    )
                )
        return citations

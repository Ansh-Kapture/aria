from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

import httpx
from bs4 import BeautifulSoup

from aria.schemas.retrieval import SearchResult

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


class WebSearchTool:
    """Web search using DuckDuckGo (free) with optional Tavily fallback.

    Implements exponential backoff on failures and extracts full page
    text for deeper chunk retrieval.
    """

    def __init__(
        self,
        tavily_api_key: str = "",
        use_tavily: bool = False,
        max_results: int = 8,
        fetch_full_text: bool = True,
    ) -> None:
        self._use_tavily = use_tavily and bool(tavily_api_key)
        self._tavily_key = tavily_api_key
        self._max_results = max_results
        self._fetch_full_text = fetch_full_text
        self._last_request_time: float = 0.0
        self._min_interval: float = 1.0  # rate limit: 1 req/sec

    async def search(self, query: str) -> list[SearchResult]:
        """Run a search and return results with optional full page text."""
        if self._use_tavily:
            results = await self._tavily_search(query)
        else:
            results = await self._ddg_search(query)

        if self._fetch_full_text:
            results = await self._enrich_with_full_text(results)

        return results

    # ── DuckDuckGo ───────────────────────────────────────────────────────────

    async def _ddg_search(self, query: str) -> list[SearchResult]:
        await self._rate_limit()
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                from duckduckgo_search import DDGS

            loop = asyncio.get_event_loop()
            raw = await loop.run_in_executor(
                None, lambda: list(DDGS().text(query, max_results=self._max_results))
            )
            return [
                SearchResult(
                    url=r.get("href", ""),
                    title=r.get("title", ""),
                    snippet=r.get("body", ""),
                )
                for r in raw
                if r.get("href")
            ]
        except Exception as exc:
            logger.warning("DuckDuckGo search failed for %r: %s", query, exc)
            return []

    # ── Tavily ────────────────────────────────────────────────────────────────

    async def _tavily_search(self, query: str) -> list[SearchResult]:
        await self._rate_limit()
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": self._tavily_key,
                        "query": query,
                        "max_results": self._max_results,
                        "include_raw_content": False,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                return [
                    SearchResult(
                        url=r["url"],
                        title=r.get("title", ""),
                        snippet=r.get("content", ""),
                    )
                    for r in data.get("results", [])
                ]
        except Exception as exc:
            logger.warning("Tavily search failed: %s — falling back to DuckDuckGo", exc)
            return await self._ddg_search(query)

    # ── Full text extraction ─────────────────────────────────────────────────

    async def _enrich_with_full_text(
        self, results: list[SearchResult]
    ) -> list[SearchResult]:
        async with httpx.AsyncClient(timeout=15, headers=_HEADERS, follow_redirects=True) as client:
            tasks = [self._fetch_page(client, r) for r in results]
            enriched = await asyncio.gather(*tasks, return_exceptions=True)

        out = []
        for r, full_text in zip(results, enriched):
            if isinstance(full_text, str):
                r = r.model_copy(update={"full_text": full_text[:8000]})
            out.append(r)
        return out

    async def _fetch_page(self, client: httpx.AsyncClient, result: SearchResult) -> str:
        for attempt in range(3):
            try:
                resp = await client.get(result.url)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                text = soup.get_text(separator=" ", strip=True)
                return " ".join(text.split())
            except Exception as exc:
                if attempt == 2:
                    logger.debug("Could not fetch %s: %s", result.url, exc)
                    return result.snippet
                await asyncio.sleep(2 ** attempt)
        return result.snippet

    async def _rate_limit(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            await asyncio.sleep(self._min_interval - elapsed)
        self._last_request_time = time.monotonic()

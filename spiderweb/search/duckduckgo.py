"""DuckDuckGo search provider using ddgs (free, no API key).

Configurable via SearchProviderConfig.extra_config: region, safesearch,
timeout, timelimit (d/w/m/y for recency), backend, page. The underlying
API does not support a sort order; use timelimit to bias toward recent results.
"""

import asyncio
from typing import Any

from spiderweb.observability.logging_config import get_logger
from spiderweb.search.base import SearchProvider, SearchResult, SearchResultBatch

logger = get_logger(__name__)


class DuckDuckGoSearchProvider(SearchProvider):
    """Search provider using DuckDuckGo via the ddgs library.

    Free to use, no API key required. Uses the DDGS client to run text
    searches and returns results as SearchResultBatch.
    """

    def __init__(
        self,
        region: str | None = None,
        safesearch: str = "moderate",
        timeout: int = 10,
        **kwargs: Any,
    ):
        """Initialize the DuckDuckGo search provider.

        Args:
            region: Region code (e.g. 'us-en', 'uk-en'). None = default.
            safesearch: 'on', 'moderate', or 'off'.
            timeout: Request timeout in seconds.
            **kwargs: Passed to DDGS (e.g. proxy).
        """
        self._region = region
        self._safesearch = safesearch
        self._timeout = timeout
        self._ddgs_kwargs = kwargs

    async def search(
        self,
        query: str,
        limit: int = 10,
        **kwargs: Any,
    ) -> SearchResultBatch:
        """Search DuckDuckGo and return results.

        Runs the synchronous DDGS.text() in a thread so the async
        interface is non-blocking.

        Args:
            query: Search query string.
            limit: Maximum number of results to return.
            **kwargs: Optional overrides. Passed to ddgs.text() where supported:
                region: e.g. "us-en", "uk-en" (default from constructor or "wt-wt").
                safesearch: "on", "moderate", or "off".
                timeout: Request timeout in seconds.
                timelimit: Filter by recency - "d" (day), "w" (week), "m" (month), "y" (year).
                backend: "auto", "duckduckgo", "bing", "brave", "google", etc.
                page: Result page (1-based).

        Returns:
            SearchResultBatch with title, href, body mapped to SearchResult.
        """
        region = kwargs.get("region", self._region)
        safesearch = kwargs.get("safesearch", self._safesearch)
        timeout = kwargs.get("timeout", self._timeout)
        timelimit = kwargs.get("timelimit")
        backend = kwargs.get("backend")
        page = kwargs.get("page", 1)

        def _run_search() -> list[dict[str, str]]:
            from ddgs import DDGS

            text_kwargs: dict[str, Any] = {
                "region": region or "wt-wt",
                "safesearch": safesearch,
                "max_results": limit,
                "page": page,
            }
            if timelimit is not None:
                text_kwargs["timelimit"] = timelimit
            if backend is not None:
                text_kwargs["backend"] = backend

            with DDGS(timeout=timeout, **self._ddgs_kwargs) as ddgs:
                gen = ddgs.text(query, **text_kwargs)
                return list(gen) if gen else []

        try:
            raw = await asyncio.to_thread(_run_search)
        except ImportError as e:
            raise ImportError(
                "ddgs is not installed. "
                "Install it with: pip install ddgs"
            ) from e

        results = []
        for i, item in enumerate(raw, 1):
            results.append(
                SearchResult(
                    url=item.get("href", ""),
                    title=item.get("title", ""),
                    description=item.get("body"),
                    snippet=item.get("body"),
                    position=i,
                    source="web",
                )
            )

        logger.debug(f"DuckDuckGo search '{query}' returned {len(results)} results")
        return SearchResultBatch(
            results=results,
            query=query,
            total=len(results),
        )

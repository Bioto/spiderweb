"""Tavily search provider.

Tavily is a search API optimized for LLM applications, providing
high-quality, relevant search results with citations.

Requires TAVILY_API_KEY environment variable.
"""

import asyncio
from typing import Any

from spiderweb.observability.logging_config import get_logger
from spiderweb.search.base import SearchProvider, SearchResult, SearchResultBatch

logger = get_logger(__name__)


class TavilySearchProvider(SearchProvider):
    """Search provider using Tavily API.

    Tavily provides high-quality search results optimized for LLM applications.
    Requires a Tavily API key (set TAVILY_API_KEY environment variable).

    Example:
        >>> provider = TavilySearchProvider(api_key="your-key")
        >>> results = await provider.search("python web scraping")
    """

    def __init__(
        self,
        api_key: str | None = None,
        **kwargs: Any,
    ):
        """Initialize Tavily search provider.

        Args:
            api_key: Tavily API key (defaults to TAVILY_API_KEY env var)
            **kwargs: Additional options passed to Tavily client
        """
        import os

        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            logger.warning(
                "Tavily API key not provided. Set TAVILY_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self._kwargs = kwargs

    async def search(
        self,
        query: str,
        limit: int = 10,
        **kwargs: Any,
    ) -> SearchResultBatch:
        """Search using Tavily API.

        Args:
            query: Search query string
            limit: Maximum number of results to return
            **kwargs: Additional Tavily options:
                search_depth: "basic" or "advanced" (default: "basic")
                include_answer: Include AI-generated answer (default: False)
                include_raw_content: Include raw HTML content (default: False)
                include_images: Include images in results (default: False)

        Returns:
            SearchResultBatch with search results

        Raises:
            ImportError: If tavily-python is not installed
            ValueError: If API key is not set
        """
        if not self.api_key:
            raise ValueError(
                "Tavily API key required. Set TAVILY_API_KEY environment variable "
                "or pass api_key to TavilySearchProvider constructor."
            )

        try:
            from tavily import TavilyClient
        except ImportError as e:
            raise ImportError(
                "tavily-python is not installed. Install it with: pip install tavily-python"
            ) from e

        def _run_search() -> dict[str, Any]:
            client = TavilyClient(api_key=self.api_key)
            search_kwargs = {
                "query": query,
                "max_results": limit,
                "search_depth": kwargs.get("search_depth", "basic"),
                "include_answer": kwargs.get("include_answer", False),
                "include_raw_content": kwargs.get("include_raw_content", False),
                "include_images": kwargs.get("include_images", False),
            }
            search_kwargs.update(self._kwargs)
            return client.search(**search_kwargs)

        try:
            response = await asyncio.to_thread(_run_search)
        except Exception as e:
            logger.error(f"Tavily search failed: {e}", exc_info=True)
            raise

        raw = response.get("results", [])[:limit]
        results = []
        for i, item in enumerate(raw, 1):
            results.append(
                SearchResult(
                    url=item.get("url", ""),
                    title=item.get("title", ""),
                    description=item.get("content"),
                    snippet=item.get("content"),
                    position=i,
                    source="tavily",
                    metadata={
                        "score": item.get("score"),
                        "raw_content": item.get("raw_content"),
                    },
                )
            )

        logger.debug(f"Tavily search '{query}' returned {len(results)} results")
        return SearchResultBatch(
            results=results,
            query=query,
            total=len(results),
        )

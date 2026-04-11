"""Firecrawl search provider (Firecrawl /v2/search).

Requires an API key: ``SPIDERWEB_FIRECRAWL_API_KEY``, ``FIRECRAWL_API_KEY``,
or ``SearchProviderConfig.extra_config["firecrawl_api_key"]``.

Install: ``pip install spiderweb[firecrawl]`` (or ``firecrawl-py``).
"""

from __future__ import annotations

import os
from typing import Any

from spiderweb.config import settings
from spiderweb.observability.logging_config import get_logger
from spiderweb.search.base import SearchProvider, SearchResult, SearchResultBatch

logger = get_logger(__name__)

# Keys passed through to Firecrawl ``SearchRequest`` (see firecrawl.v2.types.SearchRequest).
_FIRECRAWL_SEARCH_REQUEST_KEYS: frozenset[str] = frozenset({
    "sources",
    "categories",
    "tbs",
    "location",
    "ignore_invalid_urls",
    "timeout",
    "scrape_options",
    "integration",
})

def _resolve_api_key(explicit: str | None, init_kwargs: dict[str, Any]) -> str | None:
    key = init_kwargs.pop("firecrawl_api_key", None)
    if key is not None and str(key).strip():
        return str(key).strip()
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    key = init_kwargs.pop("api_key", None)
    if key is not None and str(key).strip():
        return str(key).strip()
    sk = getattr(settings, "firecrawl_api_key", None)
    if sk is not None and str(sk).strip():
        return str(sk).strip()
    env = os.environ.get("FIRECRAWL_API_KEY")
    if env and env.strip():
        return env.strip()
    return None


def _resolve_api_url(init_kwargs: dict[str, Any]) -> str:
    u = init_kwargs.pop("firecrawl_api_url", None)
    if u is not None and str(u).strip():
        return str(u).strip()
    return "https://api.firecrawl.dev"


def _extract_url(item: Any) -> str:
    url = getattr(item, "url", None)
    if url:
        return str(url)
    if hasattr(item, "metadata_dict"):
        md = item.metadata_dict
        return str(md.get("url") or md.get("source_url") or "")
    return ""


def _item_to_search_result(
    item: Any,
    position: int,
    *,
    source_label: str,
) -> SearchResult | None:
    url = _extract_url(item)
    if not url:
        return None
    title = getattr(item, "title", None) or ""
    if not isinstance(title, str):
        title = str(title) if title is not None else ""
    description = getattr(item, "description", None)
    if description is None:
        description = getattr(item, "snippet", None)
    if description is not None and not isinstance(description, str):
        description = str(description)
    category = getattr(item, "category", None)
    meta: dict[str, Any] = {}
    if category is not None:
        meta["category"] = category
    return SearchResult(
        url=url,
        title=title,
        description=description,
        snippet=description,
        position=position,
        source=source_label,
        category=str(category) if category is not None else None,
        metadata=meta,
    )


class FirecrawlSearchProvider(SearchProvider):
    """Search provider using Firecrawl's hosted search API."""

    def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
        kw = dict(kwargs)
        self._api_key = _resolve_api_key(api_key, kw)
        self._api_url = _resolve_api_url(kw)
        self._defaults: dict[str, Any] = {}
        for k in list(kw.keys()):
            if k in _FIRECRAWL_SEARCH_REQUEST_KEYS:
                self._defaults[k] = kw.pop(k)
        if kw:
            logger.debug(
                "FirecrawlSearchProvider ignoring unknown init kwargs: %s",
                list(kw.keys()),
            )
        self._client: Any = None
        self._client_key: tuple[Any, ...] | None = None

    async def _get_client(self) -> Any:
        try:
            from firecrawl import AsyncFirecrawl
        except ImportError as e:
            raise ImportError(
                "firecrawl-py is not installed. Install with: pip install spiderweb[firecrawl]"
            ) from e
        if not self._api_key:
            raise ValueError(
                "Firecrawl API key required for search. Set SPIDERWEB_FIRECRAWL_API_KEY, "
                "FIRECRAWL_API_KEY, or pass firecrawl_api_key in SearchProviderConfig.extra_config."
            )
        ck = (self._api_key, self._api_url)
        if self._client is None or self._client_key != ck:
            self._client = AsyncFirecrawl(api_key=self._api_key, api_url=self._api_url)
            self._client_key = ck
        return self._client

    def _merge_search_kwargs(self, limit: int, kwargs: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = dict(self._defaults)
        for k in _FIRECRAWL_SEARCH_REQUEST_KEYS:
            if k in kwargs and kwargs[k] is not None:
                out[k] = kwargs[k]
        lim = min(max(1, limit), 100)
        out["limit"] = lim
        if "sources" not in out or out["sources"] is None:
            out["sources"] = ["web"]
        return {k: v for k, v in out.items() if v is not None}

    async def search(
        self,
        query: str,
        limit: int = 10,
        **kwargs: Any,
    ) -> SearchResultBatch:
        if not query or not str(query).strip():
            raise ValueError("Query cannot be empty")
        fc_kwargs = self._merge_search_kwargs(limit, kwargs)
        client = await self._get_client()
        try:
            data = await client.search(query.strip(), **fc_kwargs)
        except Exception as e:
            logger.error("Firecrawl search failed: %s", e, exc_info=True)
            raise

        results: list[SearchResult] = []
        pos = 0
        web = getattr(data, "web", None) or []
        for item in web:
            pos += 1
            sr = _item_to_search_result(item, pos, source_label="firecrawl")
            if sr is not None:
                results.append(sr)
        news = getattr(data, "news", None) or []
        for item in news:
            pos += 1
            sr = _item_to_search_result(item, pos, source_label="firecrawl-news")
            if sr is not None:
                results.append(sr)

        logger.debug("Firecrawl search %r returned %s results", query, len(results))
        return SearchResultBatch(results=results, query=query, total=len(results))

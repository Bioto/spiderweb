"""SearXNG metasearch provider (self-hosted or public instance).

Configure the instance base URL via ``SPIDERWEB_SEARXNG_URL``, the ``SEARXNG_URL``
environment variable, or ``SearchProviderConfig.extra_config`` with ``searxng_url``
(or ``base_url``). Optional Bearer auth: ``SPIDERWEB_SEARXNG_API_KEY`` or
``extra_config["api_key"]``.

Calls the JSON API: ``GET {base}/search?q=...&format=json``.
"""

from __future__ import annotations

import os
from typing import Any

import aiohttp

from spiderweb.config import settings
from spiderweb.observability.logging_config import get_logger
from spiderweb.search.base import SearchProvider, SearchResult, SearchResultBatch

logger = get_logger(__name__)

# Constructor / client options — never sent as SearX query parameters.
_INSTANCE_KEYS: frozenset[str] = frozenset({"searxng_url", "base_url", "api_key", "timeout"})

# Query parameters forwarded to SearXNG (see https://docs.searxng.org/dev/search_api.html).
_SEARX_QUERY_KEYS: frozenset[str] = frozenset(
    {
        "categories",
        "engines",
        "language",
        "time_range",
        "safesearch",
        "pageno",
        "theme",
        "image_proxy",
        "no_redirect",
        "no_html",
    }
)


def _normalize_instance_base(url: str) -> str:
    u = str(url).strip().rstrip("/")
    if u.endswith("/search"):
        u = u[: -len("/search")].rstrip("/")
    return u


def _resolve_base_url(
    searxng_url: str | None,
    base_url: str | None,
) -> str | None:
    for candidate in (searxng_url, base_url):
        if candidate is not None and str(candidate).strip():
            return _normalize_instance_base(str(candidate).strip())
    sk = getattr(settings, "searxng_url", None)
    if sk is not None and str(sk).strip():
        return _normalize_instance_base(str(sk).strip())
    env = os.environ.get("SEARXNG_URL")
    if env and env.strip():
        return _normalize_instance_base(env.strip())
    return None


def _resolve_api_key(explicit: str | None) -> str | None:
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    sk = getattr(settings, "searxng_api_key", None)
    if sk is not None and str(sk).strip():
        return str(sk).strip()
    env = os.environ.get("SEARXNG_API_KEY")
    if env and env.strip():
        return env.strip()
    return None


class SearxNGSearchProvider(SearchProvider):
    """Search provider using a SearXNG instance's HTTP JSON API."""

    def __init__(
        self,
        searxng_url: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: int = 30,
        **kwargs: Any,
    ) -> None:
        self._base_url = _resolve_base_url(searxng_url, base_url)
        self._api_key = _resolve_api_key(api_key)
        self._timeout = aiohttp.ClientTimeout(total=float(timeout))
        self._default_params: dict[str, Any] = {}
        for k, v in kwargs.items():
            if k in _INSTANCE_KEYS or v is None:
                continue
            if k in _SEARX_QUERY_KEYS:
                self._default_params[k] = v
            else:
                logger.debug("SearxNGSearchProvider ignoring unknown init kwarg: %s", k)

    def _search_endpoint(self) -> str:
        if not self._base_url:
            raise ValueError(
                "SearXNG base URL is required. Set SPIDERWEB_SEARXNG_URL or SEARXNG_URL, "
                "or pass searxng_url (or base_url) in SearchProviderConfig.extra_config."
            )
        return f"{self._base_url}/search"

    def _merge_params(self, query: str, limit: int, kwargs: dict[str, Any]) -> dict[str, Any]:
        params: dict[str, Any] = {"q": query.strip(), "format": "json"}
        merged = dict(self._default_params)
        for k, v in kwargs.items():
            if k in _INSTANCE_KEYS or v is None:
                continue
            if k in _SEARX_QUERY_KEYS:
                merged[k] = v
        params.update(merged)
        return params

    async def search(
        self,
        query: str,
        limit: int = 10,
        **kwargs: Any,
    ) -> SearchResultBatch:
        if not query or not str(query).strip():
            raise ValueError("Query cannot be empty")

        url = self._search_endpoint()
        params = self._merge_params(query, limit, kwargs)
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            async with (
                aiohttp.ClientSession(timeout=self._timeout) as session,
                session.get(url, params=params, headers=headers) as resp,
            ):
                if resp.status != 200:
                    text = await resp.text()
                    logger.error(
                        "SearXNG search HTTP %s for %r: %s",
                        resp.status,
                        url,
                        text[:500],
                    )
                    if resp.status == 403:
                        raise ValueError(
                            "SearXNG search failed with HTTP 403. Common causes: "
                            "`format=json` is not enabled (add `json` under `search.formats` "
                            "in the instance settings.yml), or bot protection blocked the request."
                        )
                    raise ValueError(f"SearXNG search failed with HTTP {resp.status}")
                try:
                    payload_raw: Any = await resp.json(content_type=None)
                except aiohttp.ContentTypeError:
                    payload_raw = {}
                if not isinstance(payload_raw, dict):
                    raise ValueError("SearXNG returned a non-object JSON response")
                payload = payload_raw
        except aiohttp.ClientError as e:
            logger.error("SearXNG request failed: %s", e, exc_info=True)
            raise ValueError(f"SearXNG request failed: {e}") from e

        raw_results = payload.get("results") or []
        if not isinstance(raw_results, list):
            raw_results = []

        results: list[SearchResult] = []
        pos = 0
        for item in raw_results:
            if pos >= limit:
                break
            if not isinstance(item, dict):
                continue
            u = item.get("url") or item.get("link") or ""
            if not u:
                continue
            pos += 1
            title = item.get("title") or ""
            if not isinstance(title, str):
                title = str(title) if title is not None else ""
            desc = item.get("content")
            if desc is not None and not isinstance(desc, str):
                desc = str(desc)
            engine = item.get("engine")
            meta: dict[str, Any] = {}
            if engine is not None:
                meta["engine"] = engine
            results.append(
                SearchResult(
                    url=str(u),
                    title=title,
                    description=desc,
                    snippet=desc,
                    position=pos,
                    source="searxng",
                    metadata=meta,
                )
            )

        logger.debug("SearXNG search %r returned %s results", query, len(results))
        return SearchResultBatch(results=results, query=query, total=len(results))

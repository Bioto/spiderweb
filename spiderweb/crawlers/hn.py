"""Hacker News crawler using Firebase REST API and Algolia HN Search API.

Supports:
- Fetching single items by URL (crawl/crawl_many)
- Keyword search (search)
- User submissions (scrape_user)
"""

import asyncio
import re
from datetime import UTC, datetime
from typing import Any

import aiohttp

from spiderweb.crawlers.base import Crawler, CrawlResult
from spiderweb.models.config import CrawlerConfig, HNScraperConfig
from spiderweb.observability.logging_config import get_logger

logger = get_logger(__name__)

HN_ITEM_URL_PATTERN = re.compile(
    r"^https?://(?:www\.)?news\.ycombinator\.com/item\?id=(\d+)",
    re.IGNORECASE,
)

FIREBASE_BASE = "https://hacker-news.firebaseio.com/v0"
ALGOLIA_BASE = "https://hn.algolia.com/api/v1"


def _is_hn_item_url(url: str) -> bool:
    """Return True if the URL is an HN item URL."""
    return HN_ITEM_URL_PATTERN.match(url.strip()) is not None


def _item_id_from_url(url: str) -> str | None:
    """Extract item ID from an HN item URL, or None if not matched."""
    m = HN_ITEM_URL_PATTERN.match(url.strip())
    return m.group(1) if m else None


def _get_hn_scraper_config(config: CrawlerConfig | None) -> HNScraperConfig:
    """Resolve HNScraperConfig from CrawlerConfig.extra_config or default."""
    if config and config.extra_config.get("hn_scraper_config"):
        raw = config.extra_config["hn_scraper_config"]
        return HNScraperConfig.model_validate(raw) if isinstance(raw, dict) else raw
    return HNScraperConfig()


def _item_to_content(item: dict[str, Any]) -> str:
    """Build markdown content from an HN item dict."""
    title = item.get("title", "") or ""
    text = item.get("text", "") or ""
    url = item.get("url", "") or ""
    item_type = item.get("type", "story")

    lines = []
    if title:
        lines.append(f"# {title}")
    if url and item_type == "story":
        lines.append(f"\nLink: {url}")
    if text:
        lines.append(f"\n{text}")
    return "\n".join(lines).strip() or "(no content)"


def _item_to_crawl_result(item: dict[str, Any], url: str) -> CrawlResult:
    """Build a CrawlResult from an HN item dict."""
    item_id = str(item.get("id", ""))
    item_url = url or f"https://news.ycombinator.com/item?id={item_id}"
    item_type = item.get("type", "story")

    created_at = None
    if "created_at_i" in item:
        try:
            created_at = datetime.fromtimestamp(item["created_at_i"], tz=UTC).isoformat()
        except (TypeError, ValueError, OSError):
            pass

    content = _item_to_content(item)
    return CrawlResult(
        url=item_url,
        content=content,
        markdown=content,
        status_code=200,
        metadata={
            "source_type": f"hn_{item_type}",
            "item_id": item_id,
            "author": item.get("author", ""),
            "score": item.get("points", item.get("score", 0)),
            "num_comments": item.get("num_comments", item.get("descendants", 0)),
            "created_at": created_at,
        },
        links=[],
        success=True,
        error=None,
    )


class HNCrawler:
    """Crawler for Hacker News via Firebase and Algolia APIs.

    - crawl(url) / crawl_many(urls): fetch single items by URL.
    - search(query, config): search HN via Algolia.
    - scrape_user(handle, config): fetch a user's submissions via Algolia.
    """

    def __init__(self) -> None:
        """Initialize HN crawler."""
        self._session: aiohttp.ClientSession | None = None
        logger.debug("Initialized HNCrawler")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _fetch_item(self, item_id: str) -> dict[str, Any] | None:
        """Fetch a single HN item from Firebase API."""
        session = await self._get_session()
        url = f"{FIREBASE_BASE}/item/{item_id}.json"
        async with session.get(url) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            return data if isinstance(data, dict) and data.get("id") else None

    async def _search_algolia(
        self,
        query: str,
        tags: str = "story",
        numeric_filters: str | None = None,
        hits_per_page: int = 100,
    ) -> list[dict[str, Any]]:
        """Search HN via Algolia API (search_by_date for chronological)."""
        session = await self._get_session()
        params: dict[str, str | int] = {
            "query": query,
            "tags": tags,
            "hitsPerPage": hits_per_page,
        }
        if numeric_filters:
            params["numericFilters"] = numeric_filters

        url = f"{ALGOLIA_BASE}/search_by_date"
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()
            hits = data.get("hits", [])
            return hits if isinstance(hits, list) else []

    async def crawl(self, url: str, config: CrawlerConfig | None = None) -> CrawlResult:
        """Fetch a single HN item by URL.

        Args:
            url: An news.ycombinator.com/item?id=... URL.
            config: Optional CrawlerConfig (provider='hn').

        Returns:
            CrawlResult with item content and metadata.
        """
        if config is None:
            config = CrawlerConfig(provider="hn")

        url = url.strip()
        if not _is_hn_item_url(url):
            return CrawlResult(
                url=url,
                content="",
                status_code=0,
                success=False,
                error="URL is not an HN item URL (expected .../item?id=<id>).",
            )

        item_id = _item_id_from_url(url)
        if not item_id:
            return CrawlResult(
                url=url,
                content="",
                status_code=0,
                success=False,
                error="Could not parse item ID from URL.",
            )

        try:
            item = await self._fetch_item(item_id)
            if not item:
                return CrawlResult(
                    url=url,
                    content="",
                    status_code=404,
                    success=False,
                    error="Item not found.",
                )
            return _item_to_crawl_result(item, url)
        except Exception as e:
            logger.exception("HN API request failed for %s: %s", url, e)
            return CrawlResult(
                url=url,
                content="",
                status_code=0,
                success=False,
                error=str(e),
            )

    async def crawl_many(
        self,
        urls: list[str],
        config: CrawlerConfig | None = None,
    ) -> list[CrawlResult]:
        """Fetch multiple HN items by URL."""
        if config is None:
            config = CrawlerConfig(provider="hn")
        urls = [u for u in urls if (u or "").strip()]
        if not urls:
            return []

        hn_config = _get_hn_scraper_config(config)
        results: list[CrawlResult] = []
        for url in urls:
            result = await self.crawl(url, config)
            results.append(result)
            await asyncio.sleep(hn_config.delay_between_requests)
        return results

    async def search(
        self,
        query: str,
        config: CrawlerConfig | None = None,
    ) -> list[CrawlResult]:
        """Search HN by keywords via Algolia.

        Args:
            query: Search query.
            config: Optional CrawlerConfig; HNScraperConfig in extra_config.

        Returns:
            List of CrawlResult (one per item).
        """
        if config is None:
            config = CrawlerConfig(provider="hn")

        hn_config = _get_hn_scraper_config(config)
        type_map = {"story": "story", "comment": "comment", "all": "(story,comment)"}
        tags = type_map.get(hn_config.search_type, "story")

        try:
            hits = await self._search_algolia(
                query=query,
                tags=tags,
                hits_per_page=min(hn_config.max_results, 1000),
            )
            results: list[CrawlResult] = []
            for hit in hits[: hn_config.max_results]:
                item_id = hit.get("objectID", hit.get("id"))
                url = f"https://news.ycombinator.com/item?id={item_id}"
                item = {
                    "id": item_id,
                    "title": hit.get("title", ""),
                    "text": hit.get("story_text", hit.get("text", "")),
                    "url": hit.get("url", ""),
                    "type": hit.get("type", "story"),
                    "author": hit.get("author", ""),
                    "points": hit.get("points", 0),
                    "num_comments": hit.get("num_comments", hit.get("comment_nb", 0)),
                    "created_at_i": hit.get("created_at_i"),
                }
                results.append(_item_to_crawl_result(item, url))
                await asyncio.sleep(hn_config.delay_between_requests)
            return results
        except Exception as e:
            logger.exception("HN search failed for query '%s': %s", query, e)
            raise

    async def scrape_user(
        self,
        handle: str,
        config: CrawlerConfig | None = None,
    ) -> list[CrawlResult]:
        """Fetch a user's HN submissions via Algolia author filter.

        Args:
            handle: HN username.
            config: Optional CrawlerConfig; HNScraperConfig in extra_config.

        Returns:
            List of CrawlResult (one per submission).
        """
        if config is None:
            config = CrawlerConfig(provider="hn")

        handle = handle.strip()
        hn_config = _get_hn_scraper_config(config)

        session = await self._get_session()
        params: dict[str, str | int] = {
            "tags": f"story,author_{handle}",
            "hitsPerPage": min(hn_config.max_results, 1000),
        }

        try:
            url = f"{ALGOLIA_BASE}/search_by_date"
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                hits = data.get("hits", [])
                if not isinstance(hits, list):
                    return []

            results: list[CrawlResult] = []
            for hit in hits[: hn_config.max_results]:
                item_id = hit.get("objectID", hit.get("id"))
                item_url = f"https://news.ycombinator.com/item?id={item_id}"
                item = {
                    "id": item_id,
                    "title": hit.get("title", ""),
                    "text": hit.get("story_text", hit.get("text", "")),
                    "url": hit.get("url", ""),
                    "type": hit.get("type", "story"),
                    "author": hit.get("author", handle),
                    "points": hit.get("points", 0),
                    "num_comments": hit.get("num_comments", hit.get("comment_nb", 0)),
                    "created_at_i": hit.get("created_at_i"),
                }
                results.append(_item_to_crawl_result(item, item_url))
                await asyncio.sleep(hn_config.delay_between_requests)
            return results
        except Exception as e:
            logger.exception("HN user scrape failed for %s: %s", handle, e)
            raise

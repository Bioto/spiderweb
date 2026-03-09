"""RSS/Atom feed crawler using feedparser.

Supports:
- Fetching feed entries by feed URL (crawl/crawl_many)
- Treating feed URL as "user" for scrape_user (feed URL = handle)
"""

import asyncio
from datetime import UTC, datetime
from time import mktime, struct_time
from typing import Any

from spiderweb.crawlers.base import Crawler, CrawlResult
from spiderweb.models.config import CrawlerConfig, RSSScraperConfig
from spiderweb.observability.logging_config import get_logger

logger = get_logger(__name__)


def _get_rss_scraper_config(config: CrawlerConfig | None) -> RSSScraperConfig:
    """Resolve RSSScraperConfig from CrawlerConfig.extra_config or default."""
    if config and config.extra_config.get("rss_scraper_config"):
        raw = config.extra_config["rss_scraper_config"]
        return RSSScraperConfig.model_validate(raw) if isinstance(raw, dict) else raw
    return RSSScraperConfig()


def _published_to_iso(parsed: struct_time | None) -> str | None:
    """Convert feedparser published_parsed to ISO string."""
    if not parsed:
        return None
    try:
        dt = datetime.fromtimestamp(mktime(parsed), tz=UTC)
        return dt.isoformat()
    except (TypeError, ValueError, OSError):
        return None


def _entry_to_crawl_result(entry: Any, feed_url: str, include_summary: bool = True) -> CrawlResult:
    """Build a CrawlResult from a feedparser entry."""
    title = getattr(entry, "title", "") or ""
    link = getattr(entry, "link", "") or ""
    summary = ""
    if include_summary:
        summary = getattr(entry, "summary", "") or ""
        if not summary and hasattr(entry, "content") and entry.content:
            summary = entry.content[0].get("value", "") if isinstance(entry.content[0], dict) else ""
    body = f"# {title}\n\n{summary}".strip() if title or summary else summary or title or "(no content)"
    created_at = _published_to_iso(getattr(entry, "published_parsed", None))
    return CrawlResult(
        url=link,
        content=body,
        markdown=body,
        status_code=200,
        metadata={
            "source_type": "rss_entry",
            "feed_url": feed_url,
            "title": title,
            "created_at": created_at,
        },
        links=[],
        success=True,
        error=None,
    )


async def _parse_feed(url: str) -> dict[str, Any]:
    """Parse an RSS/Atom feed using feedparser (in thread pool)."""
    try:
        import feedparser
    except ImportError as e:
        raise ImportError(
            "feedparser is required for the RSS crawler. "
            "Install with: pip install spiderweb[rss]"
        ) from e
    return await asyncio.to_thread(feedparser.parse, url)


class RSSCrawler:
    """Crawler for RSS/Atom feeds via feedparser.

    - crawl(url): parse feed URL, return first entry as CrawlResult (protocol compliance).
    - crawl_many(urls): parse multiple feeds, return all entries as CrawlResult list.
    - scrape_user(handle, config): treat handle as feed URL, fetch and return entries.
    - search: not applicable for RSS; raises NotImplementedError.
    """

    def __init__(self) -> None:
        """Initialize RSS crawler."""
        logger.debug("Initialized RSSCrawler")

    async def crawl(self, url: str, config: CrawlerConfig | None = None) -> CrawlResult:
        """Parse a single feed URL and return the first entry.

        For protocol compliance, crawl() returns one CrawlResult. Use crawl_many
        or scrape_user to get all entries from a feed.

        Args:
            url: Feed URL (RSS or Atom).
            config: Optional CrawlerConfig (provider='rss').

        Returns:
            CrawlResult for the first entry, or empty result if feed has no entries.
        """
        if config is None:
            config = CrawlerConfig(provider="rss")

        url = url.strip()
        if not url:
            return CrawlResult(
                url=url,
                content="",
                status_code=0,
                success=False,
                error="Empty URL.",
            )

        try:
            feed = await _parse_feed(url)
            r_config = _get_rss_scraper_config(config)
            entries = getattr(feed, "entries", [])[: r_config.max_entries]
            if not entries:
                return CrawlResult(
                    url=url,
                    content="",
                    markdown="",
                    status_code=200,
                    metadata={"source_type": "rss_feed", "feed_url": url},
                    success=True,
                    error=None,
                )
            first = entries[0]
            link = getattr(first, "link", url)
            return _entry_to_crawl_result(first, url, r_config.include_summary)
        except Exception as e:
            logger.exception("RSS parse failed for %s: %s", url, e)
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
        """Parse multiple feeds and return all entries as CrawlResult list."""
        if config is None:
            config = CrawlerConfig(provider="rss")
        urls = [u for u in urls if (u or "").strip()]
        if not urls:
            return []

        r_config = _get_rss_scraper_config(config)
        results: list[CrawlResult] = []
        for feed_url in urls:
            try:
                feed = await _parse_feed(feed_url)
                entries = getattr(feed, "entries", [])[: r_config.max_entries]
                for entry in entries:
                    results.append(_entry_to_crawl_result(entry, feed_url, r_config.include_summary))
            except Exception as e:
                logger.warning("RSS parse failed for %s: %s", feed_url, e)
                results.append(
                    CrawlResult(
                        url=feed_url,
                        content="",
                        status_code=0,
                        success=False,
                        error=str(e),
                    )
                )
        return results

    async def search(
        self,
        query: str,
        config: CrawlerConfig | None = None,
    ) -> list[CrawlResult]:
        """Search is not applicable for RSS feeds.

        Raises:
            NotImplementedError: RSS feeds do not support keyword search.
        """
        raise NotImplementedError(
            "RSS crawler does not support search. "
            "Use scrape_user(feed_url) to fetch entries from a feed, "
            "or crawl_many([url1, url2, ...]) for multiple feeds."
        )

    async def scrape_user(
        self,
        handle: str,
        config: CrawlerConfig | None = None,
    ) -> list[CrawlResult]:
        """Fetch entries from a feed URL.

        The handle is treated as the feed URL (e.g. handle =
        "https://example.com/feed.xml").

        Args:
            handle: Feed URL.
            config: Optional CrawlerConfig; RSSScraperConfig in extra_config.

        Returns:
            List of CrawlResult (one per entry).
        """
        if config is None:
            config = CrawlerConfig(provider="rss")

        feed_url = handle.strip()
        if not feed_url:
            return []

        r_config = _get_rss_scraper_config(config)
        try:
            feed = await _parse_feed(feed_url)
            entries = getattr(feed, "entries", [])[: r_config.max_entries]
            return [
                _entry_to_crawl_result(entry, feed_url, r_config.include_summary)
                for entry in entries
            ]
        except Exception as e:
            logger.exception("RSS scrape failed for %s: %s", feed_url, e)
            raise

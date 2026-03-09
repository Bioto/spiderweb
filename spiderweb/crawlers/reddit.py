"""Reddit API crawler using aiohttp and Reddit OAuth2.

Supports:
- Fetching single posts by URL (crawl/crawl_many)
- Search across Reddit (search)
- User submitted posts (scrape_user)

Uses Reddit's JSON API with client_credentials OAuth (no asyncpraw).
"""

import asyncio
import base64
import re
from datetime import UTC, datetime
from typing import Any

import aiohttp

from spiderweb.config import settings
from spiderweb.crawlers.base import Crawler, CrawlResult
from spiderweb.models.config import CrawlerConfig, RedditScraperConfig
from spiderweb.observability.logging_config import get_logger

logger = get_logger(__name__)

REDDIT_OAUTH = "https://www.reddit.com/api/v1/access_token"
REDDIT_API = "https://oauth.reddit.com"

# reddit.com/r/sub/comments/id/title or old.reddit.com/...
REDDIT_SUBMISSION_PATTERN = re.compile(
    r"^https?://(?:www\.|old\.)?reddit\.com/r/[^/]+/comments/([a-z0-9]+)",
    re.IGNORECASE,
)


def _is_reddit_submission_url(url: str) -> bool:
    """Return True if the URL is a Reddit submission URL."""
    return REDDIT_SUBMISSION_PATTERN.match(url.strip()) is not None


def _submission_id_from_url(url: str) -> str | None:
    """Extract submission ID from a Reddit URL, or None if not matched."""
    m = REDDIT_SUBMISSION_PATTERN.match(url.strip())
    return m.group(1) if m else None


def _get_reddit_credentials(config: CrawlerConfig | None) -> tuple[str, str, str]:
    """Resolve Reddit credentials from config extra_config or settings."""
    extra = (config.extra_config or {}) if config else {}
    client_id = extra.get("reddit_client_id") or settings.reddit_client_id
    client_secret = extra.get("reddit_client_secret") or settings.reddit_client_secret
    user_agent = extra.get("reddit_user_agent") or settings.reddit_user_agent

    if not client_id or not client_secret:
        raise ValueError(
            "Reddit API credentials are required for the 'reddit' crawler. "
            "Set SPIDERWEB_REDDIT_CLIENT_ID and SPIDERWEB_REDDIT_CLIENT_SECRET, "
            "or pass them in CrawlerConfig.extra_config."
        )
    return client_id, client_secret, user_agent


def _get_reddit_scraper_config(config: CrawlerConfig | None) -> RedditScraperConfig:
    """Resolve RedditScraperConfig from CrawlerConfig.extra_config or default."""
    if config and config.extra_config.get("reddit_scraper_config"):
        raw = config.extra_config["reddit_scraper_config"]
        return RedditScraperConfig.model_validate(raw) if isinstance(raw, dict) else raw
    return RedditScraperConfig()


def _listing_to_crawl_result(post: dict[str, Any], url: str) -> CrawlResult:
    """Build a CrawlResult from a Reddit API post/listing child."""
    data = post.get("data", post)
    if not isinstance(data, dict):
        data = post

    title = data.get("title", "")
    selftext = data.get("selftext", "")
    body = f"# {title}\n\n{selftext}".strip() if title or selftext else selftext or title or "(link post)"

    created_utc = data.get("created_utc")
    created_at = None
    if created_utc is not None:
        created_at = datetime.fromtimestamp(created_utc, tz=UTC).isoformat()

    permalink = data.get("permalink", "")
    result_url = f"https://www.reddit.com{permalink}" if permalink else url

    return CrawlResult(
        url=result_url,
        content=body,
        markdown=body,
        status_code=200,
        metadata={
            "source_type": "reddit_submission",
            "submission_id": data.get("id", ""),
            "subreddit": data.get("subreddit", ""),
            "author": data.get("author", ""),
            "score": data.get("score", 0),
            "num_comments": data.get("num_comments", 0),
            "created_at": created_at,
        },
        links=[],
        success=True,
        error=None,
    )


class RedditCrawler:
    """Crawler for Reddit via OAuth2 and JSON API (aiohttp).

    - crawl(url) / crawl_many(urls): fetch single submissions by URL.
    - search(query, config): search across Reddit (r/all).
    - scrape_user(handle, config): fetch a user's submitted posts.
    """

    def __init__(self) -> None:
        """Initialize Reddit crawler."""
        self._session: aiohttp.ClientSession | None = None
        self._token: str | None = None
        self._token_expires: float = 0
        logger.debug("Initialized RedditCrawler")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _get_token(self, config: CrawlerConfig | None) -> str:
        """Get OAuth access token (client_credentials grant)."""
        import time

        if self._token and time.time() < self._token_expires - 60:
            return self._token

        client_id, client_secret, user_agent = _get_reddit_credentials(config)
        auth = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
        session = await self._get_session()
        async with session.post(
            REDDIT_OAUTH,
            headers={
                "Authorization": f"Basic {auth}",
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": user_agent,
            },
            data={"grant_type": "client_credentials"},
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"Reddit OAuth failed: {resp.status} {text}")
            data = await resp.json()
        self._token = data.get("access_token")
        expires_in = data.get("expires_in", 3600)
        self._token_expires = time.time() + expires_in
        if not self._token:
            raise RuntimeError("Reddit OAuth: no access_token in response")
        return self._token

    async def _request(
        self,
        path: str,
        params: dict[str, str | int] | None,
        config: CrawlerConfig | None,
    ) -> dict[str, Any]:
        """GET from oauth.reddit.com with auth."""
        token = await self._get_token(config)
        client_id, _client_secret, user_agent = _get_reddit_credentials(config)
        url = f"{REDDIT_API}{path}"
        session = await self._get_session()
        async with session.get(
            url,
            params=params or {},
            headers={
                "Authorization": f"Bearer {token}",
                "User-Agent": user_agent,
            },
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"Reddit API {resp.status}: {text[:200]}")
            return await resp.json()

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def crawl(self, url: str, config: CrawlerConfig | None = None) -> CrawlResult:
        """Fetch a single Reddit submission by URL.

        Args:
            url: A reddit.com submission URL.
            config: Optional CrawlerConfig (provider='reddit').

        Returns:
            CrawlResult with submission content and metadata.
        """
        if config is None:
            config = CrawlerConfig(provider="reddit")

        url = url.strip()
        if not _is_reddit_submission_url(url):
            return CrawlResult(
                url=url,
                content="",
                status_code=0,
                success=False,
                error="URL is not a Reddit submission URL (expected .../r/sub/comments/<id>/...).",
            )

        submission_id = _submission_id_from_url(url)
        if not submission_id:
            return CrawlResult(
                url=url,
                content="",
                status_code=0,
                success=False,
                error="Could not parse submission ID from URL.",
            )

        try:
            # /comments/article_id returns [Listing, Listing] - first is the post
            data = await self._request(f"/comments/{submission_id}", {"raw_json": 1}, config)
            if not data or not isinstance(data, list):
                return CrawlResult(
                    url=url,
                    content="",
                    status_code=404,
                    success=False,
                    error="Submission not found.",
                )
            listing = data[0]
            children = listing.get("data", {}).get("children", [])
            if not children:
                return CrawlResult(
                    url=url,
                    content="",
                    status_code=404,
                    success=False,
                    error="Submission not found.",
                )
            post = children[0]
            return _listing_to_crawl_result(post, url)
        except ValueError as e:
            logger.warning("Reddit crawler config error: %s", e)
            return CrawlResult(
                url=url,
                content="",
                status_code=0,
                success=False,
                error=str(e),
            )
        except Exception as e:
            logger.exception("Reddit API request failed for %s: %s", url, e)
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
        """Fetch multiple Reddit submissions by URL."""
        if config is None:
            config = CrawlerConfig(provider="reddit")
        urls = [u for u in urls if (u or "").strip()]
        if not urls:
            return []

        r_config = _get_reddit_scraper_config(config)
        results: list[CrawlResult] = []
        for url in urls:
            result = await self.crawl(url, config)
            results.append(result)
            await asyncio.sleep(r_config.delay_between_requests)
        return results

    async def search(
        self,
        query: str,
        config: CrawlerConfig | None = None,
    ) -> list[CrawlResult]:
        """Search Reddit by keywords (across r/all).

        Args:
            query: Search query.
            config: Optional CrawlerConfig; RedditScraperConfig in extra_config.

        Returns:
            List of CrawlResult (one per submission).
        """
        if config is None:
            config = CrawlerConfig(provider="reddit")

        r_config = _get_reddit_scraper_config(config)
        sort_map = {"new": "new", "hot": "relevance", "top": "top", "rising": "relevance", "relevance": "relevance"}
        sort_param = sort_map.get(r_config.sort, "relevance")

        try:
            data = await self._request(
                "/r/all/search",
                {
                    "q": query,
                    "sort": sort_param,
                    "t": r_config.time_filter,
                    "limit": min(r_config.max_results, 100),
                    "raw_json": 1,
                },
                config,
            )
            children = data.get("data", {}).get("children", [])
            results: list[CrawlResult] = []
            for post in children[: r_config.max_results]:
                results.append(_listing_to_crawl_result(post, ""))
                await asyncio.sleep(r_config.delay_between_requests)
            return results
        except Exception as e:
            logger.exception("Reddit search failed for query '%s': %s", query, e)
            raise

    async def scrape_user(
        self,
        handle: str,
        config: CrawlerConfig | None = None,
    ) -> list[CrawlResult]:
        """Fetch a Reddit user's submitted posts.

        Args:
            handle: Reddit username (with or without u/).
            config: Optional CrawlerConfig; RedditScraperConfig in extra_config.

        Returns:
            List of CrawlResult (one per submission).
        """
        if config is None:
            config = CrawlerConfig(provider="reddit")

        handle = handle.strip().lstrip("u/")
        r_config = _get_reddit_scraper_config(config)

        sort_map = {"new": "new", "hot": "hot", "top": "top", "rising": "rising", "relevance": "new"}
        sort_param = sort_map.get(r_config.sort, "new")

        try:
            data = await self._request(
                f"/user/{handle}/submitted",
                {
                    "sort": sort_param,
                    "t": r_config.time_filter,
                    "limit": min(r_config.max_results, 100),
                    "raw_json": 1,
                },
                config,
            )
            children = data.get("data", {}).get("children", [])
            results: list[CrawlResult] = []
            for post in children[: r_config.max_results]:
                results.append(_listing_to_crawl_result(post, ""))
                await asyncio.sleep(r_config.delay_between_requests)
            return results
        except Exception as e:
            logger.exception("Reddit user scrape failed for %s: %s", handle, e)
            raise

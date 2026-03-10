"""X (Twitter) API v2 crawler using aiohttp and Bearer token auth.

Supports:
- Fetching single tweets by status URL (crawl / crawl_many)
- Recent tweet search (search)
- User profile scraping with optional followers/following graph (scrape_user)

Requires an API v2 Bearer token (OAuth 2.0 App-only).
Set SPIDERWEB_X_BEARER_TOKEN or pass via CrawlerConfig extra_config:
    CrawlerConfig(provider="x", extra_config={"x_bearer_token": "..."})
"""

import asyncio
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import aiohttp

from spiderweb.config import settings
from spiderweb.crawlers.base import CrawlResult
from spiderweb.models.config import CrawlerConfig, XScraperConfig
from spiderweb.observability.logging_config import get_logger

logger = get_logger(__name__)

X_API = "https://api.twitter.com/2"

_X_STATUS_PATTERN = re.compile(
    r"^https?://(?:www\.)?(x\.com|twitter\.com)/[^/]+/status/(\d+)",
    re.IGNORECASE,
)


# ── Module-level helpers (also used in tests) ──────────────────────────────────

def _is_x_status_url(url: str) -> bool:
    """Return True if the URL is an X/Twitter status URL."""
    return _X_STATUS_PATTERN.match(url.strip()) is not None


def _tweet_id_from_url(url: str) -> str | None:
    """Extract the numeric tweet ID from an X/Twitter status URL, or None."""
    m = _X_STATUS_PATTERN.match(url.strip())
    return m.group(2) if m else None


def _get_x_scraper_config(config: CrawlerConfig | None) -> XScraperConfig:
    """Extract XScraperConfig from CrawlerConfig.extra_config, or return defaults."""
    if config and config.extra_config:
        raw = config.extra_config.get("x_scraper_config")
        if raw:
            if isinstance(raw, dict):
                return XScraperConfig(**raw)
            if isinstance(raw, XScraperConfig):
                return raw
    return XScraperConfig()


def _get_bearer_token(config: CrawlerConfig | None) -> str | None:
    """Resolve Bearer token from extra_config first, then global settings."""
    if config and config.extra_config:
        token = config.extra_config.get("x_bearer_token")
        if token:
            return str(token)
    return settings.x_bearer_token


# ── Result types ───────────────────────────────────────────────────────────────

@dataclass
class XUserScrapeResult:
    """Result from scraping an X user profile with optional followers/following."""

    user: dict[str, Any]
    followers: list[dict[str, Any]] = field(default_factory=list)
    following: list[dict[str, Any]] = field(default_factory=list)

    def to_crawl_results(self) -> list[CrawlResult]:
        """Convert the user profile (and optionally follower/following lists) to CrawlResults."""
        username = self.user.get("username", "")
        name = self.user.get("name", "")
        description = self.user.get("description", "")

        lines = [f"# {name} (@{username})"]
        if description:
            lines.append(f"\n{description}")
        if self.followers:
            handles = ", ".join(f["username"] for f in self.followers[:20])
            lines.append(f"\nFollowers: {handles}")
        if self.following:
            handles = ", ".join(f["username"] for f in self.following[:20])
            lines.append(f"Following: {handles}")

        return [
            CrawlResult(
                url=f"https://x.com/{username}",
                content="\n".join(lines),
                metadata={
                    "source_type": "x_user",
                    "x_user_id": self.user.get("id", ""),
                    "username": username,
                    "name": name,
                    "follower_count": len(self.followers),
                    "following_count": len(self.following),
                },
            )
        ]


# ── Crawler ────────────────────────────────────────────────────────────────────

class XCrawler:
    """X (Twitter) API v2 crawler.

    Uses aiohttp with a Bearer token. The session is created lazily on the
    first request and must be closed explicitly with `await crawler.close()`.
    """

    def __init__(self) -> None:
        self._session: aiohttp.ClientSession | None = None

    def _get_session(self, bearer_token: str) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {bearer_token}",
                    "User-Agent": "spiderweb:v0.1",
                }
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # ── Public interface ───────────────────────────────────────────────────────

    async def crawl(self, url: str, config: CrawlerConfig | None = None) -> CrawlResult:
        """Fetch a single tweet by status URL."""
        if not _is_x_status_url(url):
            return CrawlResult(
                url=url,
                content="",
                success=False,
                error=f"URL is not an X/Twitter status URL: {url}",
            )

        bearer_token = _get_bearer_token(config)
        if not bearer_token:
            return CrawlResult(
                url=url,
                content="",
                success=False,
                error=(
                    "Bearer token is required. Set SPIDERWEB_X_BEARER_TOKEN "
                    "or pass x_bearer_token in CrawlerConfig.extra_config."
                ),
            )

        tweet_id = _tweet_id_from_url(url)
        try:
            payload = await self._fetch_tweet(tweet_id, bearer_token)
            return self._tweet_payload_to_result(url, payload)
        except Exception as exc:
            logger.error("X crawl failed", extra={"url": url, "error": str(exc)})
            return CrawlResult(url=url, content="", success=False, error=str(exc))

    async def crawl_many(
        self, urls: list[str], config: CrawlerConfig | None = None
    ) -> list[CrawlResult]:
        """Fetch multiple tweets by status URL."""
        return [await self.crawl(url, config) for url in urls]

    async def search(
        self, query: str, config: CrawlerConfig | None = None
    ) -> list[CrawlResult]:
        """Search recent tweets using the X API v2 recent search endpoint."""
        bearer_token = _get_bearer_token(config)
        if not bearer_token:
            return [
                CrawlResult(
                    url=f"https://x.com/search?q={query}",
                    content="",
                    success=False,
                    error=(
                        "Bearer token is required. Set SPIDERWEB_X_BEARER_TOKEN "
                        "or pass x_bearer_token in CrawlerConfig.extra_config."
                    ),
                )
            ]

        x_config = _get_x_scraper_config(config)
        results: list[CrawlResult] = []

        for _ in range(x_config.search_max_pages):
            try:
                payload = await self._search_tweets(query, bearer_token, x_config)
            except Exception as exc:
                logger.error("X search failed", extra={"query": query, "error": str(exc)})
                break

            tweets = payload.get("data", [])
            users_by_id = {u["id"]: u for u in payload.get("includes", {}).get("users", [])}

            for tweet in tweets:
                results.append(self._tweet_data_to_result(tweet, users_by_id))
                if len(results) >= x_config.max_search_results:
                    return results

            if "next_token" not in payload.get("meta", {}):
                break

            await asyncio.sleep(x_config.delay_between_requests)

        return results

    async def scrape_user(
        self, handle: str, config: CrawlerConfig | None = None
    ) -> XUserScrapeResult:
        """Scrape a user profile, optionally including followers and following."""
        bearer_token = _get_bearer_token(config)
        x_config = _get_x_scraper_config(config)

        handle = handle.lstrip("@")
        user = await self._user_by_username(handle, bearer_token)

        followers: list[dict[str, Any]] = []
        following: list[dict[str, Any]] = []

        if x_config.include_followers:
            payload = await self._user_followers(user["id"], bearer_token, x_config)
            followers = payload.get("data", [])

        if x_config.include_following:
            payload = await self._user_following(user["id"], bearer_token, x_config)
            following = payload.get("data", [])

        return XUserScrapeResult(user=user, followers=followers, following=following)

    # ── Private API helpers ────────────────────────────────────────────────────

    async def _fetch_tweet(self, tweet_id: str, bearer_token: str) -> dict[str, Any]:
        session = self._get_session(bearer_token)
        async with session.get(
            f"{X_API}/tweets/{tweet_id}",
            params={"expansions": "author_id", "user.fields": "username,name"},
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def _search_tweets(
        self, query: str, bearer_token: str, x_config: XScraperConfig
    ) -> dict[str, Any]:
        session = self._get_session(bearer_token)
        async with session.get(
            f"{X_API}/tweets/search/recent",
            params={
                "query": query,
                "max_results": min(x_config.max_search_results, 100),
                "expansions": "author_id",
                "user.fields": "username,name",
                "tweet.fields": "created_at,author_id",
            },
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def _user_by_username(self, username: str, bearer_token: str) -> dict[str, Any]:
        session = self._get_session(bearer_token)
        async with session.get(
            f"{X_API}/users/by/username/{username}",
            params={"user.fields": "description,name,username"},
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data["data"]

    async def _user_followers(
        self, user_id: str, bearer_token: str, x_config: XScraperConfig
    ) -> dict[str, Any]:
        session = self._get_session(bearer_token)
        async with session.get(
            f"{X_API}/users/{user_id}/followers",
            params={
                "max_results": min(x_config.max_followers_per_user, 1000),
                "user.fields": "username,name,description",
            },
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def _user_following(
        self, user_id: str, bearer_token: str, x_config: XScraperConfig
    ) -> dict[str, Any]:
        session = self._get_session(bearer_token)
        async with session.get(
            f"{X_API}/users/{user_id}/following",
            params={
                "max_results": min(x_config.max_following_per_user, 1000),
                "user.fields": "username,name,description",
            },
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    # ── Data conversion ────────────────────────────────────────────────────────

    def _tweet_payload_to_result(self, url: str, payload: dict[str, Any]) -> CrawlResult:
        tweet = payload.get("data", {})
        users_by_id = {u["id"]: u for u in payload.get("includes", {}).get("users", [])}
        return self._tweet_data_to_result(tweet, users_by_id, url=url)

    def _tweet_data_to_result(
        self,
        tweet: dict[str, Any],
        users_by_id: dict[str, dict[str, Any]],
        url: str | None = None,
    ) -> CrawlResult:
        tweet_id = tweet.get("id", "")
        text = tweet.get("text", "")
        author_id = tweet.get("author_id", "")
        created_at_raw = tweet.get("created_at", "")

        author = users_by_id.get(author_id, {})
        username = author.get("username", "")
        name = author.get("name", "")

        if not url:
            url = (
                f"https://x.com/{username}/status/{tweet_id}"
                if username
                else f"https://x.com/i/status/{tweet_id}"
            )

        try:
            published_at = (
                datetime.fromisoformat(created_at_raw.replace("Z", "+00:00"))
                if created_at_raw
                else datetime.now(UTC)
            )
        except ValueError:
            published_at = datetime.now(UTC)

        content = f"# {name} (@{username})\n\n{text}"

        return CrawlResult(
            url=url,
            content=content,
            metadata={
                "source_type": "x_tweet",
                "tweet_id": tweet_id,
                "author_id": author_id,
                "username": username,
                "name": name,
                "created_at": published_at.isoformat(),
            },
        )

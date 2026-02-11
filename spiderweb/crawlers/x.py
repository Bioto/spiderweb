"""X (Twitter) API crawler using the X API v2.

Supports:
- Fetching single posts by URL (crawl/crawl_many)
- Search by keywords or hashtags (search)
- User profile and social graph: followers, following, with configurable depth (scrape_user)
"""

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

import aiohttp

from spiderweb.config import settings
from spiderweb.crawlers.base import Crawler, CrawlResult
from spiderweb.models.config import CrawlerConfig, XScraperConfig
from spiderweb.observability.logging_config import get_logger

logger = get_logger(__name__)

# x.com or twitter.com /username/status/<tweet_id>
X_STATUS_PATTERN = re.compile(
    r"^https?://(?:www\.)?(?:x\.com|twitter\.com)/[^/]+/status/(\d+)/?",
    re.IGNORECASE,
)

X_API_BASE = "https://api.x.com/2"


def _is_x_status_url(url: str) -> bool:
    """Return True if the URL is an X/Twitter status (single post) URL."""
    return X_STATUS_PATTERN.match(url.strip()) is not None


def _tweet_id_from_url(url: str) -> str | None:
    """Extract tweet ID from an X/Twitter status URL, or None if not matched."""
    m = X_STATUS_PATTERN.match(url.strip())
    return m.group(1) if m else None


def username_from_tweet_url(url: str) -> str | None:
    """Extract X username from a tweet URL (e.g. https://x.com/username/status/123)."""
    if not url:
        return None
    m = re.match(
        r"^https?://(?:www\.)?(?:x\.com|twitter\.com)/([^/]+)/status/\d+",
        url.strip(),
        re.IGNORECASE,
    )
    return m.group(1) if m else None


def _get_bearer_token(config: CrawlerConfig | None) -> str:
    """Resolve Bearer token from config extra_config or settings."""
    if config and getattr(config, "extra_config", {}).get("x_bearer_token"):
        return config.extra_config["x_bearer_token"]
    token = settings.x_bearer_token
    if not token:
        raise ValueError(
            "X API Bearer token is required for the 'x' crawler. "
            "Set SPIDERWEB_X_BEARER_TOKEN or pass extra_config['x_bearer_token'] in CrawlerConfig."
        )
    return token


def _replied_to_id(tweet: dict[str, Any]) -> str | None:
    """Return the parent tweet id if this tweet is a reply, else None."""
    refs = tweet.get("referenced_tweets") or []
    for ref in refs:
        if isinstance(ref, dict) and ref.get("type") == "replied_to":
            return ref.get("id")
    return None


def _get_x_scraper_config(config: CrawlerConfig | None) -> XScraperConfig:
    """Resolve XScraperConfig from CrawlerConfig.extra_config or default."""
    if config and config.extra_config.get("x_scraper_config"):
        raw = config.extra_config["x_scraper_config"]
        return XScraperConfig.model_validate(raw) if isinstance(raw, dict) else raw
    return XScraperConfig()


@dataclass
class XUserScrapeResult:
    """Result of scraping an X user and their social graph.

    Attributes:
        user: Resolved user object (id, username, name, description, etc.)
        followers: List of user objects for followers (up to config limits)
        following: List of user objects for accounts they follow
        levels: When graph_depth > 1, each element is a list of user objects for that level
        tweets: When include_tweets is True, CrawlResults for the user's timeline tweets
    """

    user: dict[str, Any]
    followers: list[dict[str, Any]] = field(default_factory=list)
    following: list[dict[str, Any]] = field(default_factory=list)
    levels: list[list[dict[str, Any]]] = field(default_factory=list)
    tweets: list[CrawlResult] = field(default_factory=list)

    def to_crawl_results(self, base_url: str = "https://x.com") -> list[CrawlResult]:
        """Convert to list of CrawlResult for ingestion into the pipeline."""
        results: list[CrawlResult] = []
        u = self.user
        username = u.get("username") or u.get("id", "")
        url = f"{base_url}/{username}"
        lines = [
            f"# {u.get('name', '')} (@{username})",
            "",
            u.get("description", "").strip() or "(no bio)",
            "",
            f"Profile: {url}",
            f"User ID: {u.get('id', '')}",
        ]
        if self.followers:
            lines.append("\n## Followers\n")
            for f in self.followers[:100]:  # cap for readability
                lines.append(f"- @{f.get('username', f.get('id', ''))} — {f.get('name', '')}")
        if self.following:
            lines.append("\n## Following\n")
            for f in self.following[:100]:
                lines.append(f"- @{f.get('username', f.get('id', ''))} — {f.get('name', '')}")
        content = "\n".join(lines)
        results.append(
            CrawlResult(
                url=url,
                content=content,
                markdown=content,
                status_code=200,
                metadata={
                    "source_type": "x_user",
                    "x_user_id": u.get("id"),
                    "username": username,
                },
                links=[],
                success=True,
                error=None,
                )
        )
        for tweet_cr in self.tweets:
            results.append(tweet_cr)
        for level_users in self.levels:
            for u in level_users:
                uname = u.get("username") or u.get("id", "")
                uurl = f"{base_url}/{uname}"
                ucontent = f"# {u.get('name', '')} (@{uname})\n\n{u.get('description', '')}\n\nProfile: {uurl}"
                results.append(
                    CrawlResult(
                        url=uurl,
                        content=ucontent,
                        markdown=ucontent,
                        status_code=200,
                        metadata={
                            "source_type": "x_user",
                            "x_user_id": u.get("id"),
                        },
                        links=[],
                        success=True,
                        error=None,
                    )
                )
        return results


class XCrawler:
    """Crawler for X (Twitter) via the X API v2.

    - crawl(url) / crawl_many(urls): fetch single posts by status URL.
    - search(query): search by keywords or hashtags (e.g. 'python', '#python').
    - scrape_user(username): fetch user profile, followers, and following, with optional depth.
    """

    def __init__(self) -> None:
        """Initialize X API crawler."""
        self._session: aiohttp.ClientSession | None = None
        logger.debug("Initialized XCrawler")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _request_get(
        self,
        path: str,
        params: dict[str, Any] | None,
        bearer_token: str,
        timeout_seconds: int = 30,
    ) -> dict[str, Any]:
        """GET a relative path on X API v2."""
        url = f"{X_API_BASE.rstrip('/')}/{path.lstrip('/')}"
        headers = {
            "Authorization": f"Bearer {bearer_token}",
            "Content-Type": "application/json",
        }
        session = await self._get_session()
        async with session.get(
            url,
            params=params or {},
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout_seconds),
        ) as resp:
            payload = await resp.json()
            if resp.status != 200:
                errors = payload.get("errors", [])
                msg = "; ".join(
                    e.get("detail", e.get("title", str(e))) for e in errors
                ) or resp.reason or "Unknown error"
                raise RuntimeError(f"X API error {resp.status}: {msg}")
            return payload

    async def _fetch_tweet(
        self,
        tweet_id: str,
        bearer_token: str,
        timeout_seconds: int = 30,
    ) -> dict[str, Any]:
        """Call X API GET /2/tweets/:id with optional expansions for author."""
        return await self._request_get(
            f"tweets/{tweet_id}",
            params={
                "tweet.fields": "created_at,public_metrics,author_id,conversation_id",
                "expansions": "author_id",
                "user.fields": "username,name",
            },
            bearer_token=bearer_token,
            timeout_seconds=timeout_seconds,
        )

    async def _search_tweets(
        self,
        query: str,
        max_results: int,
        next_token: str | None,
        bearer_token: str,
        timeout_seconds: int = 30,
    ) -> dict[str, Any]:
        """Call X API GET /2/tweets/search/recent. Query can be keywords or #hashtags."""
        params: dict[str, Any] = {
            "query": query.strip(),
            "max_results": min(100, max(10, max_results)),
            "tweet.fields": "created_at,public_metrics,author_id,referenced_tweets,conversation_id",
            "expansions": "author_id",
            "user.fields": "username,name",
        }
        if next_token:
            params["next_token"] = next_token
        return await self._request_get(
            "tweets/search/recent",
            params=params,
            bearer_token=bearer_token,
            timeout_seconds=timeout_seconds,
        )

    async def _user_by_username(
        self,
        username: str,
        bearer_token: str,
        timeout_seconds: int = 30,
    ) -> dict[str, Any]:
        """Call X API GET /2/users/by/username/:username."""
        path = f"users/by/username/{username.strip().lstrip('@')}"
        payload = await self._request_get(
            path,
            params={"user.fields": "id,username,name,description,public_metrics,created_at"},
            bearer_token=bearer_token,
            timeout_seconds=timeout_seconds,
        )
        return payload.get("data") or {}

    async def _user_followers(
        self,
        user_id: str,
        max_results: int,
        pagination_token: str | None,
        bearer_token: str,
        timeout_seconds: int = 30,
    ) -> dict[str, Any]:
        """Call X API GET /2/users/:id/followers."""
        params: dict[str, Any] = {
            "max_results": min(1000, max(1, max_results)),
            "user.fields": "id,username,name,description",
        }
        if pagination_token:
            params["pagination_token"] = pagination_token
        return await self._request_get(
            f"users/{user_id}/followers",
            params=params,
            bearer_token=bearer_token,
            timeout_seconds=timeout_seconds,
        )

    async def _user_following(
        self,
        user_id: str,
        max_results: int,
        pagination_token: str | None,
        bearer_token: str,
        timeout_seconds: int = 30,
    ) -> dict[str, Any]:
        """Call X API GET /2/users/:id/following."""
        params: dict[str, Any] = {
            "max_results": min(1000, max(1, max_results)),
            "user.fields": "id,username,name,description",
        }
        if pagination_token:
            params["pagination_token"] = pagination_token
        return await self._request_get(
            f"users/{user_id}/following",
            params=params,
            bearer_token=bearer_token,
            timeout_seconds=timeout_seconds,
        )

    async def _user_tweets(
        self,
        user_id: str,
        max_results: int,
        pagination_token: str | None,
        bearer_token: str,
        timeout_seconds: int = 30,
    ) -> dict[str, Any]:
        """Call X API GET /2/users/:id/tweets (user timeline)."""
        params: dict[str, Any] = {
            "max_results": min(100, max(5, max_results)),
            "tweet.fields": "created_at,public_metrics,author_id",
            "expansions": "author_id",
            "user.fields": "username,name",
        }
        if pagination_token:
            params["pagination_token"] = pagination_token
        return await self._request_get(
            f"users/{user_id}/tweets",
            params=params,
            bearer_token=bearer_token,
            timeout_seconds=timeout_seconds,
        )

    def _payload_to_content(self, payload: dict[str, Any], url: str) -> str:
        """Turn X API tweet response into markdown-like content for ingestion."""
        data = payload.get("data") or {}
        text = data.get("text", "").strip()
        tweet_id = data.get("id", "")
        created = data.get("created_at", "")
        author_id = data.get("author_id")
        users = (payload.get("includes") or {}).get("users") or []
        author = next((u for u in users if u.get("id") == author_id), None)
        username = (author or {}).get("username", "")
        name = (author or {}).get("name", "")

        lines = []
        if name or username:
            lines.append(f"**{name or username}** (@{username})\n" if username else f"**{name}**\n")
        if created:
            lines.append(f"*{created}*\n")
        lines.append("")
        lines.append(text)
        lines.append("")
        lines.append(f"Source: {url} (tweet id: {tweet_id})")
        return "\n".join(lines)

    def _tweet_to_crawl_result(
        self,
        tweet: dict[str, Any],
        users: list[dict[str, Any]],
        base_url: str = "https://x.com",
    ) -> CrawlResult:
        """Build a CrawlResult from a tweet dict and includes.users."""
        author_id = tweet.get("author_id")
        author = next((u for u in users if u.get("id") == author_id), None)
        username = (author or {}).get("username", "unknown")
        tweet_id = tweet.get("id", "")
        url = f"{base_url}/{username}/status/{tweet_id}"
        payload = {"data": tweet, "includes": {"users": users}}
        content = self._payload_to_content(payload, url)
        return CrawlResult(
            url=url,
            content=content,
            markdown=content,
            status_code=200,
            metadata={
                "source_type": "x_tweet",
                "tweet_id": tweet_id,
                "author_id": author_id,
                "created_at": tweet.get("created_at"),
            },
            links=[],
            success=True,
            error=None,
        )

    async def search(
        self,
        query: str,
        config: CrawlerConfig | None = None,
    ) -> list[CrawlResult]:
        """Search tweets by keywords or hashtags.

        Args:
            query: Search query (e.g. 'python', '#python', 'python OR #python').
            config: Optional CrawlerConfig (provider='x'); uses XScraperConfig from extra_config for limits.

        Returns:
            List of CrawlResult (one per tweet) for ingestion.
        """
        if config is None:
            config = CrawlerConfig(provider="x")
        x_config = _get_x_scraper_config(config)
        bearer_token = _get_bearer_token(config)
        timeout = getattr(config, "timeout_seconds", 30) or 30
        results: list[CrawlResult] = []
        next_token: str | None = None
        pages = 0
        seen_parent_ids: set[str] = set()
        while pages < x_config.search_max_pages:
            await asyncio.sleep(x_config.delay_between_requests)
            payload = await self._search_tweets(
                query=query,
                max_results=x_config.max_search_results,
                next_token=next_token,
                bearer_token=bearer_token,
                timeout_seconds=timeout,
            )
            data_list = payload.get("data") or []
            users = (payload.get("includes") or {}).get("users") or []
            for tweet in data_list:
                parent_id = _replied_to_id(tweet)
                if (
                    x_config.include_parent_tweet
                    and parent_id
                    and parent_id not in seen_parent_ids
                ):
                    seen_parent_ids.add(parent_id)
                    try:
                        await asyncio.sleep(x_config.delay_between_requests)
                        parent_payload = await self._fetch_tweet(
                            parent_id, bearer_token, timeout
                        )
                        parent_data = (parent_payload.get("data") or {}).copy()
                        parent_users = list((parent_payload.get("includes") or {}).get("users") or [])
                        if parent_data:
                            results.append(
                                self._tweet_to_crawl_result(parent_data, parent_users)
                            )
                    except Exception as e:
                        logger.warning(
                            "Failed to fetch parent tweet %s for reply %s: %s",
                            parent_id,
                            tweet.get("id"),
                            e,
                        )
                results.append(self._tweet_to_crawl_result(tweet, users))
            meta = payload.get("meta") or {}
            next_token = meta.get("next_token")
            pages += 1
            if not next_token:
                break
        return results

    async def scrape_user(
        self,
        username: str,
        config: CrawlerConfig | None = None,
    ) -> XUserScrapeResult:
        """Scrape a user's profile, followers, and following, with optional graph depth.

        Args:
            username: X username (with or without @).
            config: Optional CrawlerConfig (provider='x'); XScraperConfig in extra_config controls
                max_followers_per_user, max_following_per_user, graph_depth, max_users_per_level,
                include_followers, include_following, delay_between_requests.

        Returns:
            XUserScrapeResult with user, followers, following, and optionally levels (when graph_depth > 1).
        """
        if config is None:
            config = CrawlerConfig(provider="x")
        x_config = _get_x_scraper_config(config)
        bearer_token = _get_bearer_token(config)
        timeout = getattr(config, "timeout_seconds", 30) or 30
        username = username.strip().lstrip("@")

        await asyncio.sleep(x_config.delay_between_requests)
        user = await self._user_by_username(username, bearer_token, timeout)
        if not user:
            raise RuntimeError(f"User not found: {username}")

        user_id = user.get("id")
        followers: list[dict[str, Any]] = []
        following: list[dict[str, Any]] = []

        if x_config.include_followers and user_id:
            await asyncio.sleep(x_config.delay_between_requests)
            page_token: str | None = None
            while len(followers) < x_config.max_followers_per_user:
                payload = await self._user_followers(
                    user_id, x_config.max_followers_per_user, page_token, bearer_token, timeout
                )
                data_list = payload.get("data") or []
                followers.extend(data_list)
                meta = payload.get("meta") or {}
                page_token = meta.get("next_token")
                if not page_token or len(data_list) == 0:
                    break
                await asyncio.sleep(x_config.delay_between_requests)
            followers = followers[: x_config.max_followers_per_user]

        if x_config.include_following and user_id:
            await asyncio.sleep(x_config.delay_between_requests)
            page_token = None
            while len(following) < x_config.max_following_per_user:
                payload = await self._user_following(
                    user_id, x_config.max_following_per_user, page_token, bearer_token, timeout
                )
                data_list = payload.get("data") or []
                following.extend(data_list)
                meta = payload.get("meta") or {}
                page_token = meta.get("next_token")
                if not page_token or len(data_list) == 0:
                    break
                await asyncio.sleep(x_config.delay_between_requests)
            following = following[: x_config.max_following_per_user]

        tweets_crs: list[CrawlResult] = []
        if x_config.include_tweets and x_config.max_tweets_per_user > 0 and user_id:
            await asyncio.sleep(x_config.delay_between_requests)
            try:
                payload = await self._user_tweets(
                    user_id, x_config.max_tweets_per_user, None, bearer_token, timeout
                )
                data_list = payload.get("data") or []
                users_incl = (payload.get("includes") or {}).get("users") or []
                for tweet in data_list:
                    tweets_crs.append(self._tweet_to_crawl_result(tweet, users_incl))
            except Exception as e:
                logger.warning("Failed to fetch tweets for user %s: %s", username, e)

        levels: list[list[dict[str, Any]]] = []
        if x_config.graph_depth > 1 and (followers or following):
            next_level_users = (followers + following)[: x_config.max_users_per_level]
            seen_ids: set[str] = {user_id}
            for _ in range(x_config.graph_depth - 1):
                level_batch: list[dict[str, Any]] = []
                for u in next_level_users:
                    uid = u.get("id")
                    if not uid or uid in seen_ids:
                        continue
                    seen_ids.add(uid)
                    await asyncio.sleep(x_config.delay_between_requests)
                    try:
                        fol_payload = await self._user_followers(
                            uid, 100, None, bearer_token, timeout
                        )
                        level_batch.extend((fol_payload.get("data") or [])[:10])
                    except Exception as e:
                        logger.debug("Skipping follower fetch for %s: %s", uid, e)
                    if len(level_batch) >= x_config.max_users_per_level:
                        break
                if level_batch:
                    levels.append(level_batch[: x_config.max_users_per_level])
                next_level_users = level_batch
                if not next_level_users:
                    break
        return XUserScrapeResult(
            user=user,
            followers=followers,
            following=following,
            levels=levels,
            tweets=tweets_crs,
        )

    async def crawl(self, url: str, config: CrawlerConfig | None = None) -> CrawlResult:
        """Fetch a single X post by URL using the X API v2.

        Args:
            url: An x.com or twitter.com status URL (e.g. .../username/status/123).
            config: Optional crawler config; Bearer token from extra_config or settings.

        Returns:
            CrawlResult with post content as markdown-style text.
        """
        if config is None:
            config = CrawlerConfig(provider="x")

        url = url.strip()
        if not _is_x_status_url(url):
            return CrawlResult(
                url=url,
                content="",
                status_code=0,
                success=False,
                error="URL is not an X/Twitter status URL (expected .../username/status/<id>).",
            )

        tweet_id = _tweet_id_from_url(url)
        if not tweet_id:
            return CrawlResult(
                url=url,
                content="",
                status_code=0,
                success=False,
                error="Could not parse tweet ID from URL.",
            )

        try:
            bearer_token = _get_bearer_token(config)
            timeout = getattr(config, "timeout_seconds", 30) or 30
            payload = await self._fetch_tweet(tweet_id, bearer_token, timeout)
            content = self._payload_to_content(payload, url)
            data = (payload.get("data") or {})
            metadata = {
                "source_type": "x_tweet",
                "tweet_id": tweet_id,
                "author_id": data.get("author_id"),
                "created_at": data.get("created_at"),
                "public_metrics": data.get("public_metrics"),
            }
            return CrawlResult(
                url=url,
                content=content,
                markdown=content,
                status_code=200,
                metadata=metadata,
                links=[],
                success=True,
                error=None,
            )
        except ValueError as e:
            logger.warning("X crawler config error: %s", e)
            return CrawlResult(
                url=url,
                content="",
                status_code=0,
                success=False,
                error=str(e),
            )
        except Exception as e:
            logger.exception("X API request failed for %s: %s", url, e)
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
        """Fetch multiple X posts by URL.

        Only X/Twitter status URLs are fetched via the API; others are skipped
        with a failed CrawlResult and error message.
        """
        if config is None:
            config = CrawlerConfig(provider="x")
        urls = [u for u in urls if (u or "").strip()]
        if not urls:
            return []
        results: list[CrawlResult] = []
        for url in urls:
            result = await self.crawl(url, config)
            results.append(result)
        return results

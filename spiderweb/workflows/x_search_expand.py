"""Default workflow: search X for a query, take top Y tweets, expand to posters' followers/following.

Runs automatically:
  1. Search X for the given query.
  2. Take the top Y tweet results.
  3. Collect unique usernames that posted those tweets.
  4. For each user, fetch profile + followers + following (with optional graph depth).

Use via XSearchExpandWorkflow.run() or Spiderweb.x_search_and_expand().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

ProgressCallback = Callable[[str, int | None, int | None, str | None], None]

from spiderweb.crawlers.base import CrawlResult

if TYPE_CHECKING:
    from gluellm import GlueLLM
    from spiderweb.models.result import BatchIngestionResult
from spiderweb.crawlers.x import XCrawler, XUserScrapeResult, username_from_tweet_url
from spiderweb.models.config import CrawlerConfig, XScraperConfig
from spiderweb.observability.logging_config import get_logger
from spiderweb.workflows.x_query_builder import build_x_search_query

logger = get_logger(__name__)


def _unique_posters_from_tweet_results(results: list[CrawlResult]) -> list[str]:
    """Return unique author usernames from tweet CrawlResults (from URL)."""
    usernames: set[str] = set()
    for r in results:
        if not r.success:
            continue
        u = username_from_tweet_url(r.url)
        if u and u != "unknown":
            usernames.add(u)
    return list(usernames)


@dataclass
class XSearchExpandResult:
    """Result of the X search-and-expand workflow.

    Attributes:
        query: Search query that was run (may be LLM-expanded from user input).
        resolved_query: When query was expanded via LLM, the actual X API query used; else None.
        expansion_error: When expansion was attempted but failed, the error message; else None.
        tweet_results: CrawlResults from the search (up to top_tweets).
        user_results: List of (username, XUserScrapeResult) for each poster.
        all_crawl_results: Combined list of tweet CrawlResults + user profile
            CrawlResults, in order. Use for ingest_x_crawl_results().
        ingestion_result: Set when run via Spiderweb.x_search_and_expand(..., ingest=True).
    """

    query: str
    resolved_query: str | None = None
    expansion_error: str | None = None
    tweet_results: list[CrawlResult] = field(default_factory=list)
    user_results: list[tuple[str, XUserScrapeResult]] = field(default_factory=list)
    all_crawl_results: list[CrawlResult] = field(default_factory=list)
    ingestion_result: BatchIngestionResult | None = field(default=None)

    @property
    def posters(self) -> list[str]:
        """Usernames that were expanded (posters from top tweets)."""
        return [username for username, _ in self.user_results]


class XSearchExpandWorkflow:
    """Default workflow: search X → top Y tweets → expand to each poster's followers/following.

    Configure via constructor or XScraperConfig in CrawlerConfig.extra_config.
    Run with .run(crawler=..., config=...).
    """

    def __init__(
        self,
        *,
        top_tweets: int = 10,
        max_followers_per_user: int | None = None,
        max_following_per_user: int | None = None,
        graph_depth: int | None = None,
        max_users_per_level: int | None = None,
        search_max_pages: int | None = None,
        delay_between_requests: float | None = None,
    ):
        """Optional overrides for XScraperConfig used in the workflow.

        Any None value leaves the default from XScraperConfig (or from
        config.extra_config["x_scraper_config"] when run).
        """
        self.top_tweets = top_tweets
        self.max_followers_per_user = max_followers_per_user
        self.max_following_per_user = max_following_per_user
        self.graph_depth = graph_depth
        self.max_users_per_level = max_users_per_level
        self.search_max_pages = search_max_pages
        self.delay_between_requests = delay_between_requests

    def _merge_config(self, config: CrawlerConfig) -> CrawlerConfig:
        """Return a config with workflow overrides merged into extra_config x_scraper_config."""
        extra = dict(config.extra_config or {})
        raw = extra.get("x_scraper_config")
        if isinstance(raw, dict):
            x_cfg = XScraperConfig.model_validate(raw)
        else:
            x_cfg = XScraperConfig()
        overrides: dict = {}
        if self.max_followers_per_user is not None:
            overrides["max_followers_per_user"] = self.max_followers_per_user
        if self.max_following_per_user is not None:
            overrides["max_following_per_user"] = self.max_following_per_user
        if self.graph_depth is not None:
            overrides["graph_depth"] = self.graph_depth
        if self.max_users_per_level is not None:
            overrides["max_users_per_level"] = self.max_users_per_level
        if self.search_max_pages is not None:
            overrides["search_max_pages"] = self.search_max_pages
        if self.delay_between_requests is not None:
            overrides["delay_between_requests"] = self.delay_between_requests
        if overrides:
            x_cfg = x_cfg.model_copy(update=overrides)
        extra["x_scraper_config"] = x_cfg.model_dump()
        return config.model_copy(update={"extra_config": extra})

    async def run(
        self,
        query: str,
        crawler: XCrawler | None = None,
        config: CrawlerConfig | None = None,
        llm_client: "GlueLLM | None" = None,
        expand_query: bool = True,
        progress_callback: ProgressCallback | None = None,
    ) -> XSearchExpandResult:
        """Run the workflow: search → top Y tweets → expand each poster.

        Args:
            query: User search term or X search query (e.g. "msp", "#python").
            crawler: XCrawler instance (created if None).
            config: CrawlerConfig with provider="x" and x_bearer_token (and
                optional x_scraper_config). Workflow overrides (top_tweets, etc.)
                are applied on top.
            llm_client: Optional GlueLLM client. When set and expand_query is True,
                the query is rewritten into an X Search API query using operators.
            expand_query: If True and llm_client is provided, rewrite query via LLM
                before searching (default True when llm_client is set).
            progress_callback: Optional (step, current, total, detail) callback for progress UI.

        Returns:
            XSearchExpandResult with tweet_results, user_results, all_crawl_results.
            When query was expanded, resolved_query holds the actual API query used.
        """
        def report(step: str, current: int | None = None, total: int | None = None, detail: str | None = None) -> None:
            if progress_callback:
                progress_callback(step, current, total, detail)

        if config is None:
            config = CrawlerConfig(provider="x")
        config = self._merge_config(config)
        crawler = crawler or XCrawler()

        search_query = query
        resolved_query: str | None = None
        expansion_error: str | None = None
        if llm_client and expand_query:
            report("expand", detail="Expanding query with LLM...")
            try:
                search_query = await build_x_search_query(query, llm_client)
                resolved_query = search_query
                if search_query != query:
                    logger.info(
                        "XSearchExpandWorkflow: expanded query %r → %r",
                        query,
                        search_query,
                    )
            except Exception as e:
                expansion_error = str(e)
                logger.warning("XSearchExpandWorkflow: query expansion failed, using original: %s", e)

        # 1) Search
        report("search", detail="Searching X...")
        logger.info("XSearchExpandWorkflow: searching for %r", search_query)
        tweet_results = await crawler.search(search_query, config=config)
        top = tweet_results[: self.top_tweets]
        logger.info("XSearchExpandWorkflow: got %d tweets, using top %d", len(tweet_results), len(top))

        # 2) Unique posters
        posters = _unique_posters_from_tweet_results(top)
        if not posters:
            return XSearchExpandResult(
                query=query,
                resolved_query=resolved_query,
                expansion_error=expansion_error,
                tweet_results=top,
                user_results=[],
                all_crawl_results=list(top),
            )

        # 3) Scrape each poster
        user_results: list[tuple[str, XUserScrapeResult]] = []
        for i, username in enumerate(posters):
            report("users", current=i + 1, total=len(posters), detail=f"Scraping @{username}...")
            try:
                logger.info("XSearchExpandWorkflow: scraping user @%s", username)
                scrape = await crawler.scrape_user(username, config=config)
                user_results.append((username, scrape))
            except Exception as e:
                logger.warning("XSearchExpandWorkflow: skip @%s: %s", username, e)

        # 4) Combined CrawlResults for ingestion
        all_crawl_results: list[CrawlResult] = list(top)
        for _, scrape in user_results:
            all_crawl_results.extend(scrape.to_crawl_results())

        return XSearchExpandResult(
            query=query,
            resolved_query=resolved_query,
            expansion_error=expansion_error,
            tweet_results=top,
            user_results=user_results,
            all_crawl_results=all_crawl_results,
        )

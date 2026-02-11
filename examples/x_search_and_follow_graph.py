"""Example: Search X for a term, get top Y tweets, then expand to posting users' followers/following.

Two ways to do it:

  Example 1 (workflow): Use the built-in XSearchExpandWorkflow or Spiderweb.x_search_and_expand().
  Example 2 (manual):    Use XCrawler.search(), extract posters, then scrape_user() for each.

Requirements:
  - SPIDERWEB_X_BEARER_TOKEN set (or pass x_bearer_token in extra_config).
  - pip install spiderweb (aiohttp is a dependency).

Run:
  python examples/x_search_and_follow_graph.py
"""

import asyncio
import os

from spiderweb.crawlers.base import CrawlResult
from spiderweb.crawlers.x import XCrawler, XUserScrapeResult, username_from_tweet_url
from spiderweb.models.config import CrawlerConfig, XScraperConfig
from spiderweb.workflows.x_search_expand import XSearchExpandWorkflow

# --- Config (change these) ---
SEARCH_QUERY = "msp"
TOP_Y_TWEETS = 5
MAX_FOLLOWERS_PER_USER = 20
MAX_FOLLOWING_PER_USER = 20
GRAPH_DEPTH = 1  # 1 = user + their lists; 2 = also peek into those users' followers
MAX_USERS_PER_LEVEL = 5  # when depth > 1, cap users expanded per level


def _unique_posters_from_tweet_results(results: list[CrawlResult]) -> list[str]:
    """Get unique author usernames from search tweet results (from URL)."""
    usernames: set[str] = set()
    for r in results:
        if not r.success:
            continue
        u = username_from_tweet_url(r.url)
        if u and u != "unknown":
            usernames.add(u)
    return list(usernames)


def _make_config() -> CrawlerConfig:
    token = os.environ.get("SPIDERWEB_X_BEARER_TOKEN")
    if not token:
        raise ValueError("Set SPIDERWEB_X_BEARER_TOKEN or pass x_bearer_token in extra_config")
    return CrawlerConfig(
        provider="x",
        extra_config={
            "x_bearer_token": token,
            "x_scraper_config": XScraperConfig(
                max_search_results=min(100, max(10, TOP_Y_TWEETS)),
                search_max_pages=1,
                max_followers_per_user=MAX_FOLLOWERS_PER_USER,
                max_following_per_user=MAX_FOLLOWING_PER_USER,
                include_followers=True,
                include_following=True,
                graph_depth=GRAPH_DEPTH,
                max_users_per_level=MAX_USERS_PER_LEVEL,
                delay_between_requests=0.5,
            ).model_dump(),
        },
    )


# --- Example 1: Built-in workflow (one call) ---


async def example_workflow() -> None:
    """Use the default XSearchExpandWorkflow (or Spiderweb.x_search_and_expand)."""
    print("\n=== Example 1: Built-in workflow (XSearchExpandWorkflow) ===\n")
    config = _make_config()
    workflow = XSearchExpandWorkflow(
        top_tweets=TOP_Y_TWEETS,
        max_followers_per_user=MAX_FOLLOWERS_PER_USER,
        max_following_per_user=MAX_FOLLOWING_PER_USER,
        graph_depth=GRAPH_DEPTH,
        max_users_per_level=MAX_USERS_PER_LEVEL,
    )
    result = await workflow.run(SEARCH_QUERY, crawler=XCrawler(), config=config)
    print(f"Query: {result.query!r}")
    print(f"Tweets (top {TOP_Y_TWEETS}): {len(result.tweet_results)}")
    print(f"Posters expanded: {result.posters}")
    for username, scrape in result.user_results:
        u = scrape.user
        print(f"  @{username} ({u.get('name','')}): {len(scrape.followers)} followers, {len(scrape.following)} following")
    print(f"Total CrawlResults (for ingest): {len(result.all_crawl_results)}")
    print("\nOr use Spiderweb.x_search_and_expand() for the same flow + optional ingest=True.")


# --- Example 2: Manual steps (search → posters → scrape_user) ---


async def example_manual() -> None:
    """Same flow by hand: search, top Y, extract posters, scrape_user for each."""
    print("\n=== Example 2: Manual (search → top Y → posters → scrape_user) ===\n")
    config = _make_config()
    crawler = XCrawler()
    try:
        print(f"Searching X for: {SEARCH_QUERY!r}")
        tweet_results = await crawler.search(SEARCH_QUERY, config=config)
        print(f"  Got {len(tweet_results)} tweet(s)")
        top = tweet_results[:TOP_Y_TWEETS]
        posters = _unique_posters_from_tweet_results(top)
        print(f"  Top {TOP_Y_TWEETS} results → {len(posters)} unique poster(s): {posters}")
        if not posters:
            print("No posting users found.")
            return
        user_results: list[tuple[str, XUserScrapeResult]] = []
        for username in posters:
            print(f"  Scraping user: @{username} (followers + following, depth={GRAPH_DEPTH})")
            try:
                scrape = await crawler.scrape_user(username, config=config)
                user_results.append((username, scrape))
            except Exception as e:
                print(f"    Skip @{username}: {e}")
        print("\n--- Summary ---")
        for username, scrape in user_results:
            u = scrape.user
            print(f"  @{username} ({u.get('name','')}): {len(scrape.followers)} followers, {len(scrape.following)} following")
            if scrape.levels:
                print(f"    + {sum(len(l) for l in scrape.levels)} users at next level(s)")
        all_crawl_results: list[CrawlResult] = list(top)
        for _, scrape in user_results:
            all_crawl_results.extend(scrape.to_crawl_results())
        print(f"\n  Total CrawlResults: {len(all_crawl_results)} (tweets + user profiles)")
        print("  Ingest: await web.ingest_x_crawl_results(all_crawl_results, crawler_name='XCrawler')")
    finally:
        await crawler.close()


async def main() -> None:
    try:
        config = _make_config()
    except ValueError as e:
        print(e)
        return
    await example_workflow()
    await example_manual()


if __name__ == "__main__":
    asyncio.run(main())

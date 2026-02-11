"""Tests for built-in workflows."""

import pytest
from unittest.mock import AsyncMock, patch

from spiderweb.crawlers.base import CrawlResult
from spiderweb.crawlers.x import XUserScrapeResult, username_from_tweet_url
from spiderweb.models.config import CrawlerConfig
from spiderweb.workflows.x_search_expand import (
    XSearchExpandResult,
    XSearchExpandWorkflow,
    _unique_posters_from_tweet_results,
)


class TestUsernameFromTweetUrl:
    def test_extracts_username(self):
        assert username_from_tweet_url("https://x.com/john/status/123") == "john"
        assert username_from_tweet_url("https://twitter.com/foo/status/456/") == "foo"
        assert username_from_tweet_url("https://www.x.com/handle/status/789") == "handle"

    def test_returns_none_for_invalid(self):
        assert username_from_tweet_url("https://example.com") is None
        assert username_from_tweet_url("") is None


class TestUniquePostersFromTweetResults:
    def test_extracts_unique_posters(self):
        results = [
            CrawlResult(url="https://x.com/a/status/1", content="x", success=True),
            CrawlResult(url="https://x.com/b/status/2", content="y", success=True),
            CrawlResult(url="https://x.com/a/status/3", content="z", success=True),
        ]
        assert _unique_posters_from_tweet_results(results) == ["a", "b"] or set(
            _unique_posters_from_tweet_results(results)
        ) == {"a", "b"}

    def test_skips_failed(self):
        results = [
            CrawlResult(url="https://x.com/c/status/1", content="", success=False),
        ]
        assert _unique_posters_from_tweet_results(results) == []


class TestXSearchExpandWorkflow:
    async def test_run_returns_result_with_tweets_and_users_mocked(self):
        workflow = XSearchExpandWorkflow(top_tweets=2)
        config = CrawlerConfig(
            provider="x",
            extra_config={
                "x_bearer_token": "fake",
                "x_scraper_config": {"max_followers_per_user": 5, "max_following_per_user": 5},
            },
        )
        tweet_crs = [
            CrawlResult(
                url="https://x.com/alice/status/1",
                content="t1",
                success=True,
                metadata={"source_type": "x_tweet"},
            ),
            CrawlResult(
                url="https://x.com/bob/status/2",
                content="t2",
                success=True,
                metadata={"source_type": "x_tweet"},
            ),
        ]
        user_scrape = XUserScrapeResult(
            user={"id": "u1", "username": "alice", "name": "Alice"},
            followers=[],
            following=[],
        )
        with patch.object(
            workflow.__class__.__mro__[0],
            "_merge_config",
            return_value=config,
        ):
            with patch("spiderweb.workflows.x_search_expand.XCrawler") as mock_crawler_cls:
                mock_crawler = mock_crawler_cls.return_value
                mock_crawler.search = AsyncMock(return_value=tweet_crs)
                mock_crawler.scrape_user = AsyncMock(return_value=user_scrape)
                result = await workflow.run("msp", crawler=mock_crawler, config=config)
        assert isinstance(result, XSearchExpandResult)
        assert result.query == "msp"
        assert len(result.tweet_results) == 2
        assert result.posters == ["alice", "bob"]
        assert len(result.user_results) == 2
        assert len(result.all_crawl_results) >= 2

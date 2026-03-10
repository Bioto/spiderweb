"""Web crawling components for fetching and processing web content.

This module provides a pluggable architecture for web crawling with support
for multiple backends (crawl4ai, simple HTTP, etc.) and intelligent
LLM-powered structured extraction.

Built-in crawlers are automatically registered in the global crawler_registry.
Custom crawlers can be registered for use via configuration strings.
"""

from spiderweb.crawlers.base import Crawler, CrawlResult
from spiderweb.crawlers.crawl4ai import Crawl4AICrawler
from spiderweb.crawlers.extraction import CrawlExtractor
from spiderweb.crawlers.http import HttpCrawler
from spiderweb.crawlers.opensky import OpenSkyCrawler
from spiderweb.crawlers.x import XCrawler, XUserScrapeResult
from spiderweb.registry import crawler_registry

__all__ = [
    "Crawler",
    "CrawlResult",
    "HttpCrawler",
    "Crawl4AICrawler",
    "CrawlExtractor",
    "OpenSkyCrawler",
    "XCrawler",
    "XUserScrapeResult",
]

# Register built-in crawlers
# These names correspond to CrawlerConfig.provider values
crawler_registry.register("http", HttpCrawler)
crawler_registry.register("crawl4ai", Crawl4AICrawler)
crawler_registry.register("opensky", OpenSkyCrawler)
crawler_registry.register("x", XCrawler)


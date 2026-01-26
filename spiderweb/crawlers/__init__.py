"""Web crawling components for fetching and processing web content.

This module provides a pluggable architecture for web crawling with support
for multiple backends (crawl4ai, simple HTTP, etc.) and intelligent
LLM-powered structured extraction.
"""

from spiderweb.crawlers.base import Crawler, CrawlResult
from spiderweb.crawlers.crawl4ai import Crawl4AICrawler
from spiderweb.crawlers.extraction import CrawlExtractor
from spiderweb.crawlers.http import HttpCrawler

__all__ = [
    "Crawler",
    "CrawlResult",
    "HttpCrawler",
    "Crawl4AICrawler",
    "CrawlExtractor",
]


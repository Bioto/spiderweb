"""Base crawler protocol and data structures.

Defines the interface that all web crawlers must implement, following
the same pattern as extractors and vector stores.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from spiderweb.models.config import CrawlerConfig


@dataclass
class CrawlResult:
    """Result from crawling a single URL.

    Primary text is in `content`: when markdown extraction is enabled it holds
    the markdown; otherwise the raw HTML. Use `content` for downstream
    processing (synthesis, extraction, etc.). When markdown was extracted,
    `raw_html` holds the original HTML for saving or debugging.
    """

    url: str
    content: str
    markdown: str | None = None
    raw_html: str | None = None  # Original HTML when content was set to markdown
    status_code: int = 200
    metadata: dict[str, Any] = field(default_factory=dict)
    links: list[str] = field(default_factory=list)
    success: bool = True
    error: str | None = None
    
    def __post_init__(self):
        """Ensure consistency of success and error fields."""
        if self.error and self.success:
            self.success = False


@runtime_checkable
class Crawler(Protocol):
    """Protocol for web crawlers.
    
    All crawlers must implement this interface to be compatible
    with the Spiderweb crawling pipeline.
    
    Example:
        >>> crawler = HttpCrawler()
        >>> result = await crawler.crawl("https://example.com")
        >>> print(result.markdown)
    """
    
    async def crawl(self, url: str, config: CrawlerConfig | None = None) -> CrawlResult:
        """Fetch content from a single URL.
        
        Args:
            url: URL to crawl
            config: Optional crawler configuration
            
        Returns:
            CrawlResult with fetched content and metadata
            
        Raises:
            ValueError: If URL is invalid
        """
        ...
    
    async def crawl_many(
        self,
        urls: list[str],
        config: CrawlerConfig | None = None,
    ) -> list[CrawlResult]:
        """Fetch content from multiple URLs.
        
        Args:
            urls: List of URLs to crawl
            config: Optional crawler configuration
            
        Returns:
            List of CrawlResult objects
        """
        ...


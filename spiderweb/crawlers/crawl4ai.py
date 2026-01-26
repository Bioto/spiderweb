"""Crawl4AI-based web crawler with JavaScript rendering support.

Provides advanced crawling capabilities using crawl4ai including JavaScript
execution, dynamic content extraction, and intelligent link following.
"""

import asyncio
import re
from typing import Any

from spiderweb.crawlers.base import Crawler, CrawlResult
from spiderweb.models.config import CrawlerConfig
from spiderweb.observability.logging_config import get_logger

logger = get_logger(__name__)


class Crawl4AICrawler:
    """Advanced web crawler using crawl4ai.
    
    Supports JavaScript rendering, dynamic content extraction, and
    intelligent crawling strategies. Ideal for modern web applications.
    
    Example:
        >>> crawler = Crawl4AICrawler()
        >>> config = CrawlerConfig(provider="crawl4ai", wait_for_js=True)
        >>> result = await crawler.crawl("https://example.com", config)
        >>> print(result.markdown)
    """
    
    def __init__(self):
        """Initialize Crawl4AI crawler."""
        self._crawler = None
        logger.debug("Initialized Crawl4AICrawler")
    
    async def _get_crawler(self):
        """Get or create crawl4ai crawler instance.
        
        Returns:
            Crawl4AI AsyncWebCrawler instance
        """
        if self._crawler is None:
            try:
                from crawl4ai import AsyncWebCrawler
                self._crawler = AsyncWebCrawler()
                await self._crawler.__aenter__()
            except ImportError:
                raise ImportError(
                    "crawl4ai is not installed. Install it with: pip install crawl4ai"
                )
        return self._crawler
    
    async def close(self) -> None:
        """Close the crawler and cleanup resources."""
        if self._crawler is not None:
            try:
                await self._crawler.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error closing crawler: {e}")
            finally:
                self._crawler = None
    
    def _extract_links(self, html: str, base_url: str) -> list[str]:
        """Extract absolute URLs from HTML content.
        
        Args:
            html: HTML content
            base_url: Base URL for resolving relative links
            
        Returns:
            List of absolute URLs
        """
        from urllib.parse import urljoin
        
        # Simple regex to find href attributes
        href_pattern = r'href=["\'](.*?)["\']'
        hrefs = re.findall(href_pattern, html)
        
        links = []
        for href in hrefs:
            # Skip anchors, javascript, mailto, etc.
            if href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                continue
            
            # Convert to absolute URL
            absolute_url = urljoin(base_url, href)
            
            # Only include http/https URLs
            if absolute_url.startswith(('http://', 'https://')):
                links.append(absolute_url)
        
        return list(set(links))  # Deduplicate
    
    def _should_follow_link(self, url: str, config: CrawlerConfig) -> bool:
        """Check if a link should be followed based on config patterns.
        
        Args:
            url: URL to check
            config: Crawler configuration
            
        Returns:
            True if link should be followed
        """
        # Check exclude patterns first
        for pattern in config.exclude_patterns:
            if re.search(pattern, url):
                logger.debug(f"Excluding URL {url} (matches exclude pattern: {pattern})")
                return False
        
        # Check follow patterns (if specified)
        if config.follow_patterns:
            for pattern in config.follow_patterns:
                if re.search(pattern, url):
                    return True
            # If follow patterns specified but none matched
            logger.debug(f"Excluding URL {url} (no follow pattern matched)")
            return False
        
        # No patterns specified, follow all non-excluded links
        return True
    
    async def crawl(self, url: str, config: CrawlerConfig | None = None) -> CrawlResult:
        """Fetch content from a single URL with JavaScript rendering.
        
        Args:
            url: URL to crawl
            config: Optional crawler configuration
            
        Returns:
            CrawlResult with fetched content
        """
        if config is None:
            config = CrawlerConfig(provider="crawl4ai")
        
        crawler = await self._get_crawler()
        
        try:
            logger.info(f"Crawling URL with Crawl4AI: {url}")
            
            # Prepare crawl4ai parameters
            crawl_params: dict[str, Any] = {
                "url": url,
                "bypass_cache": True,
            }
            
            # Add wait_for option if JS rendering needed
            if config.wait_for_js:
                crawl_params["js_code"] = ["window.scrollTo(0, document.body.scrollHeight);"]
            
            # Add custom headers if user agent specified
            if config.user_agent:
                crawl_params["headers"] = {"User-Agent": config.user_agent}
            
            # Merge extra config
            crawl_params.update(config.extra_config)
            
            # Perform the crawl
            result = await crawler.arun(**crawl_params)
            
            # Extract content
            content = result.html or ""
            markdown = result.markdown or ""
            
            # Extract links
            links = self._extract_links(content, url)
            
            # Build metadata
            metadata: dict[str, Any] = {
                "content_length": len(content),
                "markdown_length": len(markdown),
                "links_found": len(links),
                "success": result.success,
            }
            
            # Add additional metadata from result
            if hasattr(result, "metadata") and result.metadata:
                metadata.update(result.metadata)
            
            logger.debug(
                f"Successfully crawled {url}: {len(content)} bytes HTML, "
                f"{len(markdown)} bytes markdown, {len(links)} links"
            )
            
            return CrawlResult(
                url=url,
                content=content,
                markdown=markdown if config.extract_markdown else None,
                status_code=200 if result.success else 500,
                metadata=metadata,
                links=links,
                success=result.success,
                error=None if result.success else "Crawl failed",
            )
        
        except Exception as e:
            error_msg = f"Crawl4AI error: {e}"
            logger.error(f"Failed to crawl {url}: {error_msg}", exc_info=True)
            return CrawlResult(
                url=url,
                content="",
                status_code=0,
                success=False,
                error=error_msg,
            )
    
    async def crawl_many(
        self,
        urls: list[str],
        config: CrawlerConfig | None = None,
    ) -> list[CrawlResult]:
        """Fetch content from multiple URLs with link following support.
        
        Args:
            urls: List of starting URLs to crawl
            config: Optional crawler configuration
            
        Returns:
            List of CrawlResult objects
        """
        if config is None:
            config = CrawlerConfig(provider="crawl4ai")
        
        results: list[CrawlResult] = []
        visited: set[str] = set()
        to_crawl: list[tuple[str, int]] = [(url, 1) for url in urls]  # (url, depth)
        
        semaphore = asyncio.Semaphore(config.max_concurrent)
        
        async def crawl_with_semaphore(url: str, depth: int) -> CrawlResult:
            """Crawl with concurrency limit and rate limiting."""
            async with semaphore:
                # Respect rate limiting
                if config.delay_between_requests > 0:
                    await asyncio.sleep(config.delay_between_requests)
                return await self.crawl(url, config)
        
        while to_crawl and len(results) < config.max_pages:
            # Get next batch to crawl
            batch_size = min(config.max_concurrent, config.max_pages - len(results), len(to_crawl))
            batch = to_crawl[:batch_size]
            to_crawl = to_crawl[batch_size:]
            
            # Crawl batch concurrently
            tasks = [crawl_with_semaphore(url, depth) for url, depth in batch]
            batch_results = await asyncio.gather(*tasks)
            
            # Process results
            for (url, depth), result in zip(batch, batch_results):
                if url in visited:
                    continue
                
                visited.add(url)
                results.append(result)
                
                # Add links for following if within depth limit
                if result.success and depth < config.max_depth:
                    for link in result.links:
                        if link not in visited and self._should_follow_link(link, config):
                            # Only add if we haven't hit the page limit
                            if len(results) + len(to_crawl) < config.max_pages:
                                to_crawl.append((link, depth + 1))
        
        logger.info(f"Crawled {len(results)} pages from {len(urls)} starting URLs")
        
        return results
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._get_crawler()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


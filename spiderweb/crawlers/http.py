"""Simple HTTP-based web crawler.

Provides a lightweight crawler using aiohttp for basic HTTP GET requests
without JavaScript rendering. Useful for static content and testing.
"""

import asyncio
import re
from typing import Any
from urllib.parse import urljoin

import aiohttp
from markitdown import MarkItDown

from spiderweb.crawlers.base import Crawler, CrawlResult
from spiderweb.crawlers.url_validation import validate_http_url
from spiderweb.models.config import CrawlerConfig
from spiderweb.observability.logging_config import get_logger

logger = get_logger(__name__)


class HttpCrawler:
    """Simple HTTP crawler using aiohttp.
    
    Performs basic HTTP GET requests without JavaScript rendering.
    Suitable for static content, APIs, and simple web pages.
    
    Example:
        >>> crawler = HttpCrawler()
        >>> config = CrawlerConfig(provider="http", max_depth=1)
        >>> result = await crawler.crawl("https://example.com", config)
        >>> print(result.markdown)
    """
    
    def __init__(self):
        """Initialize HTTP crawler."""
        self._session: aiohttp.ClientSession | None = None
        self._markitdown = MarkItDown()
        logger.debug("Initialized HttpCrawler")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session.
        
        Returns:
            Active aiohttp session
        """
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    def _extract_links(self, html: str, base_url: str) -> list[str]:
        """Extract absolute URLs from HTML content.
        
        Args:
            html: HTML content
            base_url: Base URL for resolving relative links
            
        Returns:
            List of absolute URLs
        """
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
        """Fetch content from a single URL.
        
        Args:
            url: URL to crawl
            config: Optional crawler configuration
            
        Returns:
            CrawlResult with fetched content
        """
        if config is None:
            config = CrawlerConfig(provider="http")

        # Basic safety/validity check (scheme/host and obvious local targets).
        url = validate_http_url(url)
        
        session = await self._get_session()
        
        headers: dict[str, str] = {}
        if config.user_agent:
            headers["User-Agent"] = config.user_agent
        
        try:
            logger.info(f"Crawling URL: {url}")
            
            async with session.get(
                url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=config.timeout_seconds),
            ) as response:
                status_code = response.status
                content = await response.text()
                
                # Extract metadata
                metadata: dict[str, Any] = {
                    "content_type": response.headers.get("Content-Type", ""),
                    "content_length": len(content),
                    "headers": dict(response.headers),
                }
                
                # Convert to markdown if requested
                markdown = None
                if config.extract_markdown:
                    try:
                        # Use markitdown to convert HTML to markdown
                        result = self._markitdown.convert_string(content)
                        markdown = result.text_content
                    except Exception as e:
                        logger.warning(f"Failed to convert HTML to markdown: {e}")
                        markdown = content  # Fallback to raw content
                
                # Extract links for potential following
                links = self._extract_links(content, url)
                
                logger.debug(f"Successfully crawled {url}: {status_code}, {len(content)} bytes, {len(links)} links")
                
                return CrawlResult(
                    url=url,
                    content=content,
                    markdown=markdown,
                    status_code=status_code,
                    metadata=metadata,
                    links=links,
                    success=True,
                    error=None,
                )
        
        except asyncio.TimeoutError:
            error_msg = f"Timeout after {config.timeout_seconds}s"
            logger.error(f"Failed to crawl {url}: {error_msg}")
            return CrawlResult(
                url=url,
                content="",
                status_code=0,
                success=False,
                error=error_msg,
            )
        
        except aiohttp.ClientError as e:
            error_msg = f"HTTP client error: {e}"
            logger.error(f"Failed to crawl {url}: {error_msg}")
            return CrawlResult(
                url=url,
                content="",
                status_code=0,
                success=False,
                error=error_msg,
            )
        
        except Exception as e:
            error_msg = f"Unexpected error: {e}"
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
            config = CrawlerConfig(provider="http")
        
        results: list[CrawlResult] = []
        visited: set[str] = set()
        to_crawl: list[tuple[str, int]] = [(url, 1) for url in urls]  # (url, depth)
        
        semaphore = asyncio.Semaphore(config.max_concurrent)
        
        async def crawl_with_semaphore(url: str, depth: int) -> CrawlResult:
            """Crawl with concurrency limit."""
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
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


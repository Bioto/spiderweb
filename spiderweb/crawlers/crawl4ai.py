"""Crawl4AI-based web crawler with JavaScript rendering support.

Provides advanced crawling capabilities using crawl4ai including JavaScript
execution, dynamic content extraction, and intelligent link following.
"""

import asyncio
import re
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from spiderweb.crawlers.base import Crawler, CrawlResult
from spiderweb.crawlers.url_validation import validate_http_url
from spiderweb.models.config import CrawlerConfig
from spiderweb.observability.logging_config import get_logger

logger = get_logger(__name__)

# Default User-Agent for PDF fetches; SEC.gov and similar sites require a non-empty,
# browser-like User-Agent or they return 403 Forbidden.
_DEFAULT_PDF_USER_AGENT = (
    "Mozilla/5.0 (compatible; Spiderweb/1.0; research crawler; +https://github.com)"
)


def _is_sec_gov_url(url: str) -> bool:
    """Return True if the URL is from SEC.gov (which requires a proper User-Agent)."""
    try:
        return "sec.gov" in (urlparse(url).netloc or "").lower()
    except Exception:
        return False


async def _download_pdf_with_user_agent(
    url: str,
    user_agent: str,
    timeout: int = 120,
) -> str:
    """Download PDF from URL with User-Agent to a temp file; return file:// URL.

    crawl4ai's PDF strategy uses requests.get() with no headers, so SEC.gov returns
    403. For SEC (and similar) we download ourselves then pass a file:// URL.
    """
    import urllib.request

    def _get() -> str:
        req = urllib.request.Request(url, headers={"User-Agent": user_agent})
        fd, path = tempfile.mkstemp(suffix=".pdf")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                with open(fd, "wb") as f:
                    f.write(resp.read())
            return path
        except Exception:
            Path(path).unlink(missing_ok=True)
            raise

    path = await asyncio.to_thread(_get)
    return f"file://{path}"


def _is_pdf_url(url: str) -> bool:
    """Return True if the URL is likely a PDF (path or query ends with .pdf)."""
    parsed = urlparse(url)
    path = (parsed.path or "").rstrip("/").lower()
    query = (parsed.query or "").lower()
    return path.endswith(".pdf") or ".pdf" in query


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
    
    def _build_run_config(self, config: CrawlerConfig) -> Any:
        """Build crawl4ai CrawlerRunConfig from Spiderweb CrawlerConfig.
        
        Maps existing CrawlerConfig fields to crawl4ai's CrawlerRunConfig,
        then merges extra_config to allow power users to set any CrawlerRunConfig
        field. See crawl4ai's CrawlerRunConfig for full list of available options.
        
        Args:
            config: Spiderweb CrawlerConfig
            
        Returns:
            crawl4ai CrawlerRunConfig instance
        """
        try:
            from crawl4ai import CrawlerRunConfig, CacheMode
        except ImportError:
            raise ImportError(
                "crawl4ai is not installed. Install it with: pip install crawl4ai"
            )
        
        # Build base config dict from CrawlerConfig fields
        run_config_dict: dict[str, Any] = {
            # Map timeout_seconds (seconds) to page_timeout (milliseconds)
            "page_timeout": config.timeout_seconds * 1000,
            # Map respect_robots_txt to check_robots_txt
            "check_robots_txt": config.respect_robots_txt,
            # Map wait_for_js to js_code (current behavior: scroll when JS enabled)
            "js_code": ["window.scrollTo(0, document.body.scrollHeight);"] if config.wait_for_js else None,
            # Preserve current behavior: bypass cache
            "cache_mode": CacheMode.BYPASS,
        }
        
        # Map user_agent only if provided (crawl4ai's run config has user_agent field)
        if config.user_agent:
            run_config_dict["user_agent"] = config.user_agent
        
        # Map headers and cookies if provided
        if config.headers:
            run_config_dict["headers"] = config.headers
        if config.cookies:
            # crawl4ai expects cookies as a dict or CookieJar
            run_config_dict["cookies"] = config.cookies
        
        # Merge extra_config (allows power users to override or add any CrawlerRunConfig field)
        run_config_dict.update(config.extra_config)
        
        # Build and return CrawlerRunConfig
        return CrawlerRunConfig.from_kwargs(run_config_dict)

    async def _crawl_pdf(self, url: str, config: CrawlerConfig) -> CrawlResult:
        """Crawl a PDF URL using crawl4ai's PDF strategies; returns CrawlResult."""
        try:
            from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
            from crawl4ai.processors.pdf import (
                PDFContentScrapingStrategy,
                PDFCrawlerStrategy,
            )
        except ImportError as e:
            logger.warning("crawl4ai PDF strategies not available: %s. Falling back to normal crawl.", e)
            return await self._crawl_page(url, config)

        pdf_crawler_strategy = PDFCrawlerStrategy()
        pdf_scraping_strategy = PDFContentScrapingStrategy()
        run_config_dict: dict[str, Any] = {
            "page_timeout": config.timeout_seconds * 1000,
            "cache_mode": CacheMode.BYPASS,
            "scraping_strategy": pdf_scraping_strategy,
            # SEC.gov and similar sites return 403 without a proper User-Agent
            "user_agent": config.user_agent or _DEFAULT_PDF_USER_AGENT,
        }
        run_config_dict.update(config.extra_config)
        run_config = CrawlerRunConfig.from_kwargs(run_config_dict)

        user_agent = config.user_agent or _DEFAULT_PDF_USER_AGENT
        crawl_url = url
        temp_path: str | None = None

        if _is_sec_gov_url(url):
            # crawl4ai's PDF strategy uses requests.get() with no headers; SEC returns 403.
            # Download with User-Agent ourselves and pass file:// so their strategy skips fetch.
            try:
                crawl_url = await _download_pdf_with_user_agent(url, user_agent)
                temp_path = crawl_url[7:] if crawl_url.startswith("file://") else None
                logger.info("Downloaded SEC.gov PDF to temp file for extraction")
            except Exception as e:
                logger.warning("SEC PDF download with User-Agent failed: %s", e)
                return CrawlResult(
                    url=url,
                    content="",
                    raw_html=None,
                    status_code=0,
                    success=False,
                    error=f"Failed to download PDF (SEC requires User-Agent): {e}",
                )

        try:
            logger.info("Crawling PDF with Crawl4AI: %s", url)
            async with AsyncWebCrawler(crawler_strategy=pdf_crawler_strategy) as pdf_crawler:
                result = await pdf_crawler.arun(url=crawl_url, config=run_config)
            if temp_path:
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except OSError as e:
                    logger.debug("Could not remove temp PDF %s: %s", temp_path, e)
            markdown = result.markdown or ""
            html = result.html or ""
            out_content = markdown if config.extract_markdown else (markdown or html)
            metadata: dict[str, Any] = {
                "content_length": len(out_content),
                "links_found": 0,
                "success": result.success,
            }
            if hasattr(result, "metadata") and result.metadata:
                metadata.update(result.metadata)
            return CrawlResult(
                url=url,
                content=out_content,
                markdown=markdown if config.extract_markdown else None,
                raw_html=html or None,
                status_code=200 if result.success else 500,
                metadata=metadata,
                links=[],
                success=result.success,
                error=None if result.success else "PDF crawl failed",
            )
        except Exception as e:
            if temp_path:
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except OSError:
                    pass
            error_msg = f"Crawl4AI PDF error: {e}"
            logger.error("Failed to crawl PDF %s: %s", url, error_msg, exc_info=True)
            return CrawlResult(
                url=url,
                content="",
                raw_html=None,
                status_code=0,
                success=False,
                error=error_msg,
            )

    async def _crawl_page(self, url: str, config: CrawlerConfig) -> CrawlResult:
        """Fetch a single HTML page (shared logic for non-PDF crawl)."""
        crawler = await self._get_crawler()
        run_config = self._build_run_config(config)
        result = await crawler.arun(url=url, config=run_config)
        html = result.html or ""
        markdown = result.markdown or ""
        links = self._extract_links(html, url)
        if config.extract_markdown and markdown:
            out_content = markdown
            raw_html = html
        else:
            out_content = html
            raw_html = None
        metadata: dict[str, Any] = {
            "content_length": len(out_content),
            "links_found": len(links),
            "success": result.success,
        }
        if raw_html is not None:
            metadata["raw_html_length"] = len(html)
        if hasattr(result, "metadata") and result.metadata:
            metadata.update(result.metadata)
        return CrawlResult(
            url=url,
            content=out_content,
            markdown=markdown if config.extract_markdown else None,
            raw_html=raw_html,
            status_code=200 if result.success else 500,
            metadata=metadata,
            links=links,
            success=result.success,
            error=None if result.success else "Crawl failed",
        )

    async def crawl(self, url: str, config: CrawlerConfig | None = None) -> CrawlResult:
        """Fetch content from a single URL with JavaScript rendering or PDF extraction.
        
        For URLs that look like PDFs (path or query ends with .pdf), uses crawl4ai's
        PDF strategies to extract text. Otherwise uses the default page crawler.
        
        Args:
            url: URL to crawl
            config: Optional crawler configuration
            
        Returns:
            CrawlResult with fetched content
        """
        if config is None:
            config = CrawlerConfig(provider="crawl4ai")

        url = validate_http_url(url)

        if _is_pdf_url(url):
            return await self._crawl_pdf(url, config)

        try:
            return await self._crawl_page(url, config)
        except Exception as e:
            error_msg = f"Crawl4AI error: {e}"
            logger.error("Failed to crawl %s: %s", url, error_msg, exc_info=True)
            return CrawlResult(
                url=url,
                content="",
                raw_html=None,
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
        
        # Skip empty or whitespace URLs so we never call crawl("")
        urls = [u for u in urls if (u or "").strip()]
        if not urls:
            return []

        results: list[CrawlResult] = []
        visited: set[str] = set()
        to_crawl: list[tuple[str, int]] = [(u.strip(), 1) for u in urls]  # (url, depth)
        
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


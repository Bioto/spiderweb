"""Firecrawl API-based web crawler.

Uses the hosted Firecrawl scrape API via ``firecrawl-py`` (``AsyncFirecrawl``).
Requires an API key: set ``SPIDERWEB_FIRECRAWL_API_KEY``, ``FIRECRAWL_API_KEY``,
or ``CrawlerConfig.extra_config["firecrawl_api_key"]``.

Install: ``pip install spiderweb[firecrawl]`` (or ``firecrawl-py``).
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any
from urllib.parse import urljoin, urlparse

from spiderweb.config import settings
from spiderweb.crawlers.base import CrawlResult
from spiderweb.crawlers.url_validation import validate_http_url
from spiderweb.models.config import CrawlerConfig
from spiderweb.observability.logging_config import get_logger

logger = get_logger(__name__)

_FIRECRAWL_RESERVED_EXTRA_KEYS: frozenset[str] = frozenset({"firecrawl_api_key", "firecrawl_api_url"})


def _registered_domain(url: str) -> str:
    """Return a coarse eTLD+1 (e.g. 'example.com') for a URL, or netloc if unavailable."""
    netloc = urlparse(url).netloc.lower().split(":")[0]
    parts = netloc.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return netloc


def _is_pdf_url(url: str) -> bool:
    """Return True if the URL is likely a PDF (path or query ends with .pdf)."""
    parsed = urlparse(url)
    path = (parsed.path or "").rstrip("/").lower()
    query = (parsed.query or "").lower()
    return path.endswith(".pdf") or ".pdf" in query


def _extract_links_from_html(html: str, base_url: str) -> list[str]:
    """Extract absolute http(s) URLs from HTML href attributes."""
    href_pattern = r'href=["\'](.*?)["\']'
    hrefs = re.findall(href_pattern, html)
    links: list[str] = []
    for href in hrefs:
        if href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue
        absolute_url = urljoin(base_url, href)
        if absolute_url.startswith(("http://", "https://")):
            links.append(absolute_url)
    return list(dict.fromkeys(links))


def _should_follow_link(url: str, config: CrawlerConfig) -> bool:
    for pattern in config.exclude_patterns:
        if re.search(pattern, url):
            logger.debug("Excluding URL %s (matches exclude pattern: %s)", url, pattern)
            return False
    if config.follow_patterns:
        for pattern in config.follow_patterns:
            if re.search(pattern, url):
                return True
        logger.debug("Excluding URL %s (no follow pattern matched)", url)
        return False
    return True


def _resolve_api_key(config: CrawlerConfig) -> str | None:
    extra = config.extra_config or {}
    key = extra.get("firecrawl_api_key")
    if key is not None and str(key).strip():
        return str(key).strip()
    sk = getattr(settings, "firecrawl_api_key", None)
    if sk is not None and str(sk).strip():
        return str(sk).strip()
    env = os.environ.get("FIRECRAWL_API_KEY")
    if env and env.strip():
        return env.strip()
    return None


def _resolve_api_url(config: CrawlerConfig) -> str:
    extra = config.extra_config or {}
    u = extra.get("firecrawl_api_url")
    if u is not None and str(u).strip():
        return str(u).strip()
    return "https://api.firecrawl.dev"


def _firecrawl_extra_options(config: CrawlerConfig) -> dict[str, Any]:
    """Return extra_config entries to pass to scrape(), excluding Spiderweb-reserved keys."""
    extra = dict(config.extra_config or {})
    for k in _FIRECRAWL_RESERVED_EXTRA_KEYS:
        extra.pop(k, None)
    return extra


def _build_scrape_kwargs(config: CrawlerConfig, *, want_links: bool) -> dict[str, Any]:
    """Build keyword arguments for AsyncFirecrawl.scrape (maps to Firecrawl ScrapeOptions)."""
    timeout_ms = int(config.timeout_seconds * 1000)
    kwargs: dict[str, Any] = {
        "timeout": timeout_ms,
        "block_ads": config.block_ads,
    }
    if config.headers:
        kwargs["headers"] = dict(config.headers)
    if config.excluded_tags:
        kwargs["exclude_tags"] = list(config.excluded_tags)
    if config.css_selector:
        sel = config.css_selector.strip()
        # Firecrawl include_tags expects tag names (e.g. main, article), not full CSS.
        if re.match(r"^[a-zA-Z][\w:-]*$", sel):
            kwargs["include_tags"] = [sel]
    need_links = want_links or config.max_depth > 1
    kwargs["formats"] = ["markdown", "html", "links"] if need_links else ["markdown", "html"]

    kwargs.update(_firecrawl_extra_options(config))

    if config.wait_for_js:
        # Best-effort: extra wait for dynamic content (milliseconds); extra_config may override.
        kwargs.setdefault("wait_for", 500)

    return {k: v for k, v in kwargs.items() if v is not None}


def _build_scrape_kwargs_for_url(url: str, config: CrawlerConfig, *, want_links: bool) -> dict[str, Any]:
    kwargs = _build_scrape_kwargs(config, want_links=want_links)
    if _is_pdf_url(url):
        existing = kwargs.get("parsers")
        if isinstance(existing, list):
            if "pdf" not in existing:
                kwargs["parsers"] = [*existing, "pdf"]
        else:
            kwargs["parsers"] = ["pdf"]
    return kwargs


def _document_to_crawl_result(url: str, doc: Any, config: CrawlerConfig) -> CrawlResult:
    markdown = (getattr(doc, "markdown", None) or "") or ""
    html = (getattr(doc, "html", None) or getattr(doc, "raw_html", None) or "") or ""

    links_attr = getattr(doc, "links", None)
    links = [str(x) for x in links_attr if x] if isinstance(links_attr, list) else []
    if not links and html:
        links = _extract_links_from_html(html, url)

    if config.extract_markdown and markdown:
        content = markdown
        raw_html = html or None
        md_out: str | None = markdown
    else:
        content = html or markdown
        raw_html = None
        md_out = markdown if markdown else None

    metadata: dict[str, Any] = {}
    if hasattr(doc, "metadata_dict"):
        metadata.update(doc.metadata_dict)
    else:
        md = getattr(doc, "metadata", None)
        if md is not None and hasattr(md, "model_dump"):
            metadata.update(md.model_dump(exclude_none=True))

    status_code = 200
    md_obj = getattr(doc, "metadata", None)
    if md_obj is not None:
        sc = getattr(md_obj, "status_code", None)
        if sc is not None:
            status_code = int(sc)

    metadata.setdefault("content_length", len(content))
    metadata.setdefault("links_found", len(links))

    return CrawlResult(
        url=url,
        content=content,
        markdown=md_out,
        raw_html=raw_html,
        status_code=status_code,
        metadata=metadata,
        links=links,
        success=True,
        error=None,
    )


class FirecrawlCrawler:
    """Web crawler using the Firecrawl hosted scrape API."""

    def __init__(self, crawler_config: CrawlerConfig | None = None):
        self._crawler_config = crawler_config
        self._client: Any = None
        self._client_key: tuple[Any, ...] | None = None
        logger.debug("Initialized FirecrawlCrawler")

    async def _get_client(self, config: CrawlerConfig) -> Any:
        try:
            from firecrawl import AsyncFirecrawl
        except ImportError as e:
            raise ImportError(
                "firecrawl-py is not installed. Install with: pip install spiderweb[firecrawl]"
            ) from e

        api_key = _resolve_api_key(config)
        api_url = _resolve_api_url(config)
        timeout = float(config.overall_timeout_seconds)
        ck = (api_key, api_url, timeout)
        if self._client is None or self._client_key != ck:
            self._client = AsyncFirecrawl(api_key=api_key, api_url=api_url, timeout=timeout)
            self._client_key = ck
        return self._client

    async def close(self) -> None:
        self._client = None
        self._client_key = None

    async def crawl(self, url: str, config: CrawlerConfig | None = None) -> CrawlResult:
        if config is None:
            config = CrawlerConfig(provider="firecrawl")
        url = validate_http_url(url)
        want_links = config.max_depth > 1
        scrape_kwargs = _build_scrape_kwargs_for_url(url, config, want_links=want_links)
        try:
            client = await self._get_client(config)
            async with asyncio.timeout(config.overall_timeout_seconds):
                doc = await client.scrape(url, **scrape_kwargs)
            return _document_to_crawl_result(url, doc, config)
        except TimeoutError:
            msg = f"Overall timeout ({config.overall_timeout_seconds}s) exceeded"
            logger.warning("Firecrawl timeout for %s", url)
            return CrawlResult(
                url=url,
                content="",
                raw_html=None,
                status_code=0,
                success=False,
                error=msg,
            )
        except Exception as e:
            error_msg = f"Firecrawl error: {e}"
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
        if config is None:
            config = CrawlerConfig(provider="firecrawl")

        urls = [u for u in urls if (u or "").strip()]
        if not urls:
            return []

        allowed_domains: frozenset[str] = frozenset()
        if config.restrict_to_start_domains and config.max_depth > 1:
            allowed_domains = frozenset(d for u in urls if (d := _registered_domain(u.strip())))

        def _should_follow_scoped(link_url: str) -> bool:
            if allowed_domains and _registered_domain(link_url) not in allowed_domains:
                return False
            return _should_follow_link(link_url, config)

        results: list[CrawlResult] = []
        visited: set[str] = set()
        to_crawl: list[tuple[str, int]] = [(u.strip(), 1) for u in urls]
        semaphore = asyncio.Semaphore(config.max_concurrent)

        async def crawl_with_semaphore(u: str, _depth: int) -> CrawlResult:
            async with semaphore:
                if config.delay_between_requests > 0:
                    await asyncio.sleep(config.delay_between_requests)
                return await self.crawl(u, config)

        while to_crawl and len(results) < config.max_pages:
            batch_size = min(config.max_concurrent, config.max_pages - len(results), len(to_crawl))
            batch = to_crawl[:batch_size]
            to_crawl = to_crawl[batch_size:]
            tasks = [crawl_with_semaphore(u, d) for u, d in batch]
            batch_results = await asyncio.gather(*tasks)

            for (u, depth), result in zip(batch, batch_results, strict=True):
                if u in visited:
                    continue
                visited.add(u)
                results.append(result)
                if result.success and depth < config.max_depth:
                    for link in result.links:
                        if (
                            link not in visited
                            and _should_follow_scoped(link)
                            and len(results) + len(to_crawl) < config.max_pages
                        ):
                            to_crawl.append((link, depth + 1))

        logger.info("Firecrawl: crawled %s pages from %s starting URLs", len(results), len(urls))
        return results

    async def __aenter__(self) -> FirecrawlCrawler:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

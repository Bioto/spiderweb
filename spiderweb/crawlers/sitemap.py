"""Sitemap parsing for crawl URL discovery.

Parses sitemap.xml files to discover URLs for crawling.
"""

import xml.etree.ElementTree as ET
from typing import Any
from urllib.parse import urljoin, urlparse

import aiohttp

from spiderweb.observability.logging_config import get_logger

logger = get_logger(__name__)


async def parse_sitemap(url: str, session: aiohttp.ClientSession | None = None) -> list[str]:
    """Parse a sitemap.xml URL and return all URLs found.

    Supports both regular sitemaps and sitemap index files.
    Recursively follows sitemap index files to discover all URLs.

    Args:
        url: URL to sitemap.xml or sitemap index
        session: Optional aiohttp session (creates one if not provided)

    Returns:
        List of URLs found in the sitemap(s)

    Example:
        >>> urls = await parse_sitemap("https://example.com/sitemap.xml")
        >>> print(f"Found {len(urls)} URLs")
    """
    close_session = False
    if session is None:
        session = aiohttp.ClientSession()
        close_session = True

    try:
        urls = await _parse_sitemap_recursive(url, session, seen=set())
        logger.info(f"Parsed sitemap {url}: found {len(urls)} URLs")
        return urls
    finally:
        if close_session:
            await session.close()


async def _parse_sitemap_recursive(
    url: str,
    session: aiohttp.ClientSession,
    seen: set[str],
) -> list[str]:
    """Recursively parse sitemap, following sitemap index files.

    Args:
        url: Sitemap URL to parse
        session: aiohttp session
        seen: Set of already-seen sitemap URLs (to avoid cycles)

    Returns:
        List of URLs
    """
    if url in seen:
        logger.debug(f"Skipping already-seen sitemap: {url}")
        return []

    seen.add(url)

    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
            if response.status != 200:
                logger.warning(f"Sitemap {url} returned status {response.status}")
                return []

            content = await response.text()
            root = ET.fromstring(content)

            # Check if this is a sitemap index
            if root.tag.endswith("sitemapindex"):
                return await _parse_sitemap_index(root, url, session, seen)

            # Regular sitemap with URLs
            return _parse_urlset(root)
    except Exception as e:
        logger.error(f"Failed to parse sitemap {url}: {e}", exc_info=True)
        return []


async def _parse_sitemap_index(
    root: ET.Element,
    base_url: str,
    session: aiohttp.ClientSession,
    seen: set[str],
) -> list[str]:
    """Parse a sitemap index file and follow all referenced sitemaps.

    Args:
        root: XML root element of sitemap index
        base_url: Base URL for resolving relative sitemap URLs
        session: aiohttp session
        seen: Set of seen sitemap URLs

    Returns:
        Combined list of URLs from all referenced sitemaps
    """
    urls = []
    namespace = _get_namespace(root)

    for sitemap_elem in root.findall(f".//{namespace}sitemap"):
        loc_elem = sitemap_elem.find(f"{namespace}loc")
        if loc_elem is not None and loc_elem.text:
            sitemap_url = urljoin(base_url, loc_elem.text.strip())
            nested_urls = await _parse_sitemap_recursive(sitemap_url, session, seen)
            urls.extend(nested_urls)

    return urls


def _parse_urlset(root: ET.Element) -> list[str]:
    """Parse a URL set from a sitemap.

    Args:
        root: XML root element containing URLs

    Returns:
        List of URLs
    """
    urls = []
    namespace = _get_namespace(root)

    for url_elem in root.findall(f".//{namespace}url"):
        loc_elem = url_elem.find(f"{namespace}loc")
        if loc_elem is not None and loc_elem.text:
            url_str = loc_elem.text.strip()
            if url_str:
                urls.append(url_str)

    return urls


def _get_namespace(root: ET.Element) -> str:
    """Extract namespace from XML root element.

    Args:
        root: XML root element

    Returns:
        Namespace prefix (e.g., "{http://www.sitemaps.org/schemas/sitemap/0.9}")
    """
    # If tag has explicit namespace (e.g. "{http://...}localname"), use it first
    if "}" in root.tag:
        return root.tag.split("}")[0] + "}"

    # Try common sitemap namespaces
    namespaces = [
        "{http://www.sitemaps.org/schemas/sitemap/0.9}",
        "{http://www.sitemaps.org/schemas/sitemap/0.9/}",
        "",
    ]

    for ns in namespaces:
        if root.tag.startswith(ns):
            return ns

    return ""


def discover_sitemap_url(base_url: str) -> str:
    """Discover sitemap URL from a base URL.

    Common locations:
    - /sitemap.xml
    - /sitemap_index.xml
    - robots.txt (sitemap directive)

    Args:
        base_url: Base URL (e.g., https://example.com)

    Returns:
        Sitemap URL or empty string if not found
    """
    parsed = urlparse(base_url)
    base = f"{parsed.scheme}://{parsed.netloc}"

    # Common sitemap locations
    candidates = [
        f"{base}/sitemap.xml",
        f"{base}/sitemap_index.xml",
        f"{base}/sitemap-index.xml",
    ]

    # Return first candidate (caller should verify it exists)
    return candidates[0]

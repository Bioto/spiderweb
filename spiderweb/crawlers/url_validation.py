"""URL validation helpers for crawlers.

These checks are meant to prevent obvious unsafe / invalid URLs from being
fetched (e.g., non-HTTP schemes, loopback/private IP literals).

Note: This is not a perfect SSRF defense for hostnames that resolve to private
IPs via DNS. For high-security environments, add allowlists and DNS/IP
resolution checks at the network boundary.
"""

from __future__ import annotations

import ipaddress
from typing import TYPE_CHECKING

from urllib.parse import urlparse

if TYPE_CHECKING:
    import aiohttp

from spiderweb.observability.logging_config import get_logger

_logger = get_logger(__name__)


def validate_http_url(url: str) -> str:
    """Validate a URL for HTTP crawling.

    Args:
        url: Input URL.

    Returns:
        The original URL (trimmed) if valid.

    Raises:
        ValueError: If the URL is invalid or uses a disallowed scheme/host.
    """
    if not url or not url.strip():
        raise ValueError("URL must be a non-empty string")

    url = url.strip()
    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme!r} (only http/https allowed)")

    if not parsed.netloc:
        raise ValueError("URL must include a host")

    # Disallow credentials in URLs (common SSRF primitive and easy to mishandle).
    if parsed.username or parsed.password:
        raise ValueError("URLs with embedded credentials are not allowed")

    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL must include a valid hostname")

    hostname_lower = hostname.lower()
    if hostname_lower in {"localhost"} or hostname_lower.endswith(".localhost"):
        raise ValueError("Refusing to crawl localhost URLs")

    # Block IP literals that are obviously local/private.
    try:
        ip = ipaddress.ip_address(hostname)
    except ValueError:
        ip = None

    if ip is not None:
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise ValueError(f"Refusing to crawl private or non-routable IP address: {hostname}")

    return url


async def validate_url_liveness(
    url: str,
    timeout: float = 5.0,
    session: "aiohttp.ClientSession | None" = None,
) -> tuple[bool, int | None, str | None]:
    """Check URL reachability with HTTP HEAD.

    Returns:
        (is_probably_live, http_status_or_none, error_detail)

    ``is_probably_live`` is False only for clearly gone responses (404, 410, 451)
    or invalid URLs. Network errors, 405 on HEAD, and most other 4xx return True
    so the full crawler can still attempt the page.
    """
    try:
        url = validate_http_url(url)
    except ValueError as e:
        return False, None, str(e)

    import aiohttp

    close_session = False
    if session is None:
        session = aiohttp.ClientSession()
        close_session = True

    timeout_cfg = aiohttp.ClientTimeout(total=timeout)
    headers = {"User-Agent": "SpiderwebUrlCheck/1.0"}

    try:
        async with session.head(
            url,
            allow_redirects=True,
            timeout=timeout_cfg,
            headers=headers,
        ) as resp:
            status = resp.status
            if status in (404, 410, 451):
                return False, status, f"HTTP {status}"
            if status == 405:
                # Many sites disallow HEAD; allow crawl.
                return True, status, None
            if status >= 400 and status not in (401, 403, 429):
                # Ambiguous client/server errors — still try full crawl.
                return True, status, None
            return True, status, None
    except aiohttp.ClientResponseError as e:
        if e.status in (404, 410, 451):
            return False, e.status, str(e)
        return True, e.status, None
    except (aiohttp.ClientError, TimeoutError, OSError) as e:
        _logger.debug("URL liveness check inconclusive for %s: %s", url, e)
        return True, None, None
    finally:
        if close_session:
            await session.close()


async def partition_urls_by_liveness(
    urls: list[str],
    timeout: float = 5.0,
) -> tuple[list[str], list[tuple[str, str]]]:
    """HEAD-check each URL sharing one ClientSession.

    Returns:
        (urls_to_crawl, list of (url, reason) for clearly dead URLs)
    """
    import aiohttp

    live: list[str] = []
    dead: list[tuple[str, str]] = []

    async with aiohttp.ClientSession() as session:
        for u in urls:
            ok, _status, err = await validate_url_liveness(u, timeout=timeout, session=session)
            if ok:
                live.append(u)
            else:
                dead.append((u, err or "dead"))

    return live, dead


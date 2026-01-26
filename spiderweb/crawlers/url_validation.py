"""URL validation helpers for crawlers.

These checks are meant to prevent obvious unsafe / invalid URLs from being
fetched (e.g., non-HTTP schemes, loopback/private IP literals).

Note: This is not a perfect SSRF defense for hostnames that resolve to private
IPs via DNS. For high-security environments, add allowlists and DNS/IP
resolution checks at the network boundary.
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse


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




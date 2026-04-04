"""Domain-based filtering of search results."""

from __future__ import annotations

from urllib.parse import urlparse

from spiderweb.search.base import SearchResult
from spiderweb.search.trace import FilteredCandidate


def hostname_matches_domain(hostname: str, domain: str) -> bool:
    """True if hostname equals domain or is a subdomain of it."""
    host = (hostname or "").lower().rstrip(".")
    dom = domain.lower().lstrip(".")
    if not host or not dom:
        return False
    return host == dom or host.endswith("." + dom)


def url_blocked_by_domains(url: str, blocked_domains: list[str]) -> bool:
    """Return True if URL's host matches any blocked domain suffix."""
    if not blocked_domains:
        return False
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
    except Exception:
        return False
    return any(hostname_matches_domain(host, d) for d in blocked_domains)


def partition_results_by_blocked_domains(
    results: list[SearchResult],
    blocked_domains: list[str],
    source_query: str | None = None,
) -> tuple[list[SearchResult], list[FilteredCandidate]]:
    """Split results into allowed vs blocked by domain."""
    if not blocked_domains:
        return list(results), []

    allowed: list[SearchResult] = []
    filtered: list[FilteredCandidate] = []
    for r in results:
        if url_blocked_by_domains(r.url, blocked_domains):
            filtered.append(
                FilteredCandidate(
                    url=r.url,
                    title=r.title,
                    snippet=r.description or r.snippet,
                    source_query=source_query,
                    source_position=r.position,
                    filter_reason="blocked_domain",
                )
            )
        else:
            allowed.append(r)
    return allowed, filtered


def sort_results_by_preferred_domains(
    results: list[SearchResult],
    preferred_domains: list[str],
) -> list[SearchResult]:
    """Stable sort: preferred domains first (earlier in list = higher priority)."""
    if not preferred_domains or not results:
        return list(results)

    def rank(r: SearchResult) -> tuple[int, int]:
        try:
            host = (urlparse(r.url).hostname or "").lower()
        except Exception:
            host = ""
        for i, dom in enumerate(preferred_domains):
            if hostname_matches_domain(host, dom):
                return (0, i)
        return (1, len(preferred_domains))

    return sorted(results, key=rank)

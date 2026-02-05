"""Path and filename helpers for safe filesystem use."""

import re


def sanitize_query_for_path(query: str, max_length: int = 120) -> str:
    """Turn a search query into a safe directory/filename segment.

    Lowercases, replaces whitespace and unsafe chars with underscores,
    and truncates to avoid overly long paths.

    Args:
        query: Raw search query (e.g. "Python web scraping").
        max_length: Maximum length of the returned string.

    Returns:
        Safe string suitable for use in paths (e.g. "python_web_scraping").
    """
    if not query or not query.strip():
        return "query"
    s = query.strip().lower()
    s = re.sub(r"[\s]+", "_", s)
    s = re.sub(r"[^\w\-.]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        return "query"
    return s[:max_length] if len(s) > max_length else s

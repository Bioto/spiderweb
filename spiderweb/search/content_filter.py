"""Post-crawl content checks: stale signals and required patterns."""

from __future__ import annotations

import re
from typing import Any

from spiderweb.observability.logging_config import get_logger
from spiderweb.research.models import ListingItem

logger = get_logger(__name__)


def _normalize_llm_regex_pattern(p: str) -> str:
    """Fix common LLM mistakes so patterns still compile.

    Models often emit ``(?i>word`` meaning case-insensitive ``word``; in Python
    the valid form is ``(?i)word``. We already compile with IGNORECASE, so the
    fix is to drop the bogus prefix and keep the rest.
    """
    s = (p or "").strip()
    if len(s) >= 5 and (s.startswith("(?i>") or s.startswith("(?I>")):
        return s[4:].strip() or s
    return s


def _compile_patterns(patterns: list[str]) -> list[re.Pattern[str]]:
    compiled: list[re.Pattern[str]] = []
    for raw in patterns:
        if not raw or not str(raw).strip():
            continue
        p = _normalize_llm_regex_pattern(str(raw))
        if not p:
            continue
        try:
            compiled.append(re.compile(p, re.IGNORECASE | re.DOTALL))
        except re.error as e:
            logger.warning("Invalid regex in content filter, skipping %r: %s", raw, e)
    return compiled


def detect_stale_content(
    content: str,
    url: str,
    patterns: list[str],
) -> tuple[bool, str | None]:
    """Return (is_stale, first_matching_pattern) if any pattern matches."""
    if not patterns:
        return False, None
    text = content or ""
    for pat in _compile_patterns(patterns):
        if pat.search(text):
            return True, pat.pattern
    return False, None


def passes_required_patterns(
    content: str,
    patterns: list[str],
) -> tuple[bool, str | None]:
    """If patterns is empty, pass. Otherwise at least one must match.

    Returns (passed, reason_if_failed).
    """
    if not patterns:
        return True, None
    text = content or ""
    compiled = _compile_patterns(patterns)
    if not compiled:
        return True, None
    for pat in compiled:
        if pat.search(text):
            return True, None
    return False, "required_content_patterns: no match"


def filter_stale_listings(
    listings: list[ListingItem],
    stale_status_values: list[str] | None = None,
) -> list[ListingItem]:
    """Drop listings whose status substring-matches stale values (case-insensitive)."""
    if stale_status_values is None:
        stale_status_values = ["sold", "ended", "expired", "pending", "closed", "completed"]
    out: list[ListingItem] = []
    stale_lower = [s.lower() for s in stale_status_values if s]
    for item in listings:
        status = (item.status or "").lower()
        if status and any(s in status for s in stale_lower):
            continue
        out.append(item)
    return out


def page_content_for_filtering(crawl_result: Any) -> str:
    """Extract plain text for regex checks from a CrawlResult-like object."""
    if crawl_result is None:
        return ""
    md = getattr(crawl_result, "markdown", None) or ""
    body = getattr(crawl_result, "content", None) or ""
    return f"{md}\n{body}".strip()

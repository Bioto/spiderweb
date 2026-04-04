"""Apply agent SearchStrategy to search query strings."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spiderweb.research.models import SearchStrategy


def augment_search_query(base_query: str, strategy: "SearchStrategy | None") -> str:
    """Append exclusions and site restrictions from strategy to the query."""
    if not strategy:
        return base_query
    parts = [base_query.strip()]
    for x in strategy.query_exclusions:
        x = (x or "").strip()
        if x:
            parts.append(x)
    for s in strategy.query_site_restrictions:
        s = (s or "").strip()
        if s:
            parts.append(s)
    return " ".join(parts).strip()

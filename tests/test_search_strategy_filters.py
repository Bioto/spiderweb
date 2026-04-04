"""Tests for agent-controlled search strategy helpers."""

from spiderweb.research.models import ListingItem, SearchStrategy
from spiderweb.search.base import SearchResult
from spiderweb.search.content_filter import (
    detect_stale_content,
    filter_stale_listings,
    passes_required_patterns,
)
from spiderweb.search.domain_filter import (
    partition_results_by_blocked_domains,
    sort_results_by_preferred_domains,
    url_blocked_by_domains,
)
from spiderweb.search.strategy_query import augment_search_query


def test_url_blocked_by_domains():
    assert url_blocked_by_domains("https://en.wikipedia.org/wiki/Foo", ["wikipedia.org"])
    assert not url_blocked_by_domains("https://example.com/", ["wikipedia.org"])


def test_partition_results_by_blocked_domains():
    results = [
        SearchResult(url="https://en.wikipedia.org/wiki/X", title="w"),
        SearchResult(url="https://dealer.example/car", title="c"),
    ]
    allowed, filtered = partition_results_by_blocked_domains(
        results,
        ["wikipedia.org"],
        source_query="q",
    )
    assert len(allowed) == 1
    assert allowed[0].url.endswith("dealer.example/car")
    assert len(filtered) == 1
    assert filtered[0].filter_reason == "blocked_domain"


def test_sort_results_by_preferred_domains():
    results = [
        SearchResult(url="https://b.com/1", title="b"),
        SearchResult(url="https://a.com/preferred", title="a"),
    ]
    out = sort_results_by_preferred_domains(results, ["a.com"])
    assert out[0].url == "https://a.com/preferred"


def test_augment_search_query():
    s = SearchStrategy(
        query_exclusions=["-wikipedia"],
        query_site_restrictions=["site:example.org"],
    )
    assert "porsche" in augment_search_query("porsche", s)
    assert "-wikipedia" in augment_search_query("porsche", s)
    assert "site:example.org" in augment_search_query("porsche", s)


def test_detect_stale_content():
    stale, pat = detect_stale_content(
        "This listing has been removed.",
        "https://x.com",
        [r"removed"],
    )
    assert stale and pat


def test_detect_stale_content_normalizes_llm_i_gt_typo():
    """LLMs often emit (?i>manual instead of (?i)manual; we normalize."""
    stale, _ = detect_stale_content(
        "Specs: MANUAL transmission",
        "https://x.com",
        ["(?i>manual"],
    )
    assert stale


def test_passes_required_patterns():
    assert passes_required_patterns("hello world", []) == (True, None)
    assert passes_required_patterns("has KEY word", [r"KEY"])[0] is True
    assert passes_required_patterns("nope", [r"KEY"])[0] is False


def test_filter_stale_listings_drops_sold():
    items = [
        ListingItem(url="https://a.com/1", status="active"),
        ListingItem(url="https://a.com/2", status="sold"),
    ]
    out = filter_stale_listings(items)
    assert len(out) == 1
    assert out[0].url == "https://a.com/1"


def test_filter_stale_listings_keeps_unknown_status():
    items = [ListingItem(url="https://a.com/1", status=None), ListingItem(url="https://a.com/2", status="unknown")]
    out = filter_stale_listings(items)
    assert len(out) == 2

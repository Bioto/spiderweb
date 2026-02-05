"""Trace models for search-crawl sessions.

Captures the full flow of search → crawl operations including summaries,
URLs, provenance (how we got to each page), and filtered-out candidates.
"""

from dataclasses import dataclass, field
from typing import Any

from spiderweb.crawlers.base import CrawlResult
from spiderweb.search.base import SearchResultBatch


@dataclass
class PageRecord:
    """Record of a crawled page with summary and provenance.
    
    Tracks where we crawled, what we found, and how we got there
    (which search query, position, parent URL if followed from link).
    """
    
    url: str
    summary: str | None = None  # LLM-generated or truncated summary
    crawl_result: CrawlResult | None = None  # Reference to full crawl result
    source_query: str | None = None  # Which search query produced this URL
    source_position: int | None = None  # Rank in that search (1-based)
    parent_url: str | None = None  # If followed from another page; else None
    extracted_data: Any = None  # Optional structured extraction result
    links_found: list[str] = field(default_factory=list)  # Outbound URLs discovered


@dataclass
class FilteredCandidate:
    """Record of a URL/link that was filtered out (not crawled).
    
    Tracks what was excluded and why, so the dataset is complete
    and we can analyze filtering decisions.
    """
    
    url: str
    title: str | None = None
    snippet: str | None = None
    source_query: str | None = None
    source_position: int | None = None
    parent_url: str | None = None
    filtered_out: bool = True  # Always True for FilteredCandidate
    filter_reason: str = "unknown"  # e.g., "relevance", "duplicate", etc.
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchRound:
    """One round of search → crawl → summarize.
    
    Represents a single search query execution, the URLs found,
    which ones were crawled (PageRecord), and which were filtered out.
    """
    
    query: str
    search_results: SearchResultBatch
    pages: list[PageRecord] = field(default_factory=list)
    filtered_out: list[FilteredCandidate] = field(default_factory=list)
    round_number: int = 1


@dataclass
class SearchCrawlTrace:
    """Complete trace of a search-crawl session.
    
    Captures the full flow: original query, all rounds executed,
    which URLs were crawled vs filtered, summaries, and provenance.
    This is the "web" Spider-Man leaves behind — queryable knowledge
    of where he's been and how he got there.
    """
    
    original_query: str
    rounds: list[SearchRound] = field(default_factory=list)
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    
    def add_round(self, round_data: SearchRound) -> None:
        """Add a search round to the trace.
        
        Args:
            round_data: SearchRound to add
        """
        round_data.round_number = len(self.rounds) + 1
        self.rounds.append(round_data)
    
    def get_all_urls(self) -> list[str]:
        """Get all URLs that were crawled.
        
        Returns:
            List of crawled URLs
        """
        urls = []
        for round_data in self.rounds:
            for page in round_data.pages:
                urls.append(page.url)
        return urls
    
    def get_all_filtered_urls(self) -> list[str]:
        """Get all URLs that were filtered out.
        
        Returns:
            List of filtered URLs
        """
        urls = []
        for round_data in self.rounds:
            for filtered in round_data.filtered_out:
                urls.append(filtered.url)
        return urls

"""Web search components for finding and discovering URLs.

This module provides a pluggable architecture for web search with support
for multiple backends (Firecrawl, Serper, Tavily, etc.) and integration
with the crawling pipeline.
"""

from spiderweb.search.base import SearchProvider, SearchResult, SearchResultBatch
from spiderweb.registry import search_provider_registry
from spiderweb.search.duckduckgo import DuckDuckGoSearchProvider
from spiderweb.search.stub import StubSearchProvider

__all__ = [
    "SearchProvider",
    "SearchResult",
    "SearchResultBatch",
    "search_provider_registry",
    "DuckDuckGoSearchProvider",
    "StubSearchProvider",
]

# Register DuckDuckGo as default (free, no API key)
search_provider_registry.register("duckduckgo", DuckDuckGoSearchProvider)
# Register stub for testing/development
search_provider_registry.register("stub", StubSearchProvider)

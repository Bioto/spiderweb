"""Web search components for finding and discovering URLs.

This module provides a pluggable architecture for web search with support
for multiple backends (Firecrawl, SearXNG, Tavily, etc.) and integration
with the crawling pipeline.
"""

from spiderweb.registry import search_provider_registry
from spiderweb.search.base import SearchProvider, SearchResult, SearchResultBatch
from spiderweb.search.duckduckgo import DuckDuckGoSearchProvider
from spiderweb.search.searxng import SearxNGSearchProvider
from spiderweb.search.stub import StubSearchProvider

__all__ = [
    "SearchProvider",
    "SearchResult",
    "SearchResultBatch",
    "search_provider_registry",
    "DuckDuckGoSearchProvider",
    "SearxNGSearchProvider",
    "StubSearchProvider",
]

# Register DuckDuckGo as default (free, no API key)
search_provider_registry.register("duckduckgo", DuckDuckGoSearchProvider)
search_provider_registry.register("searxng", SearxNGSearchProvider)
# Register stub for testing/development
search_provider_registry.register("stub", StubSearchProvider)

# Register Tavily (requires TAVILY_API_KEY)
try:
    from spiderweb.search.tavily import TavilySearchProvider

    search_provider_registry.register("tavily", TavilySearchProvider)
    __all__.append("TavilySearchProvider")
except ImportError:
    # Tavily not available (tavily-python not installed)
    pass

# Register Firecrawl search (requires firecrawl-py + API key at runtime)
try:
    from spiderweb.search.firecrawl import FirecrawlSearchProvider

    search_provider_registry.register("firecrawl", FirecrawlSearchProvider)
    __all__.append("FirecrawlSearchProvider")
except ImportError:
    pass

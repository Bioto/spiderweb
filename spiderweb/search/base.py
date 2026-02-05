"""Base search provider protocol and data structures.

Defines the interface that all search providers must implement, using
an abstract base class to enforce the "must override" requirement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchResult:
    """A single search result with URL and metadata.
    
    Represents one URL found by a search query, including title,
    description, and optional metadata like position or category.
    """
    
    url: str
    title: str
    description: str | None = None
    snippet: str | None = None  # Alias for description, kept for compatibility
    position: int | None = None  # Rank in search results (1-based)
    source: str | None = None  # Source/category (e.g., "web", "news", "github")
    category: str | None = None  # Category if applicable (e.g., "github", "research")
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set snippet from description if not provided."""
        if self.snippet is None and self.description is not None:
            self.snippet = self.description
        if self.description is None and self.snippet is not None:
            self.description = self.snippet


@dataclass
class SearchResultBatch:
    """Batch of search results from a single query.
    
    Container for multiple SearchResult objects, optionally including
    the query that produced them and total count if available.
    """
    
    results: list[SearchResult]
    query: str | None = None
    total: int | None = None  # Total results available (may be > len(results))
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set total from results length if not provided."""
        if self.total is None:
            self.total = len(self.results)


class SearchProvider(ABC):
    """Abstract base class for web search providers.
    
    All search providers must implement the search() method. This is
    enforced via ABC + @abstractmethod, ensuring subclasses cannot be
    instantiated without overriding search.
    
    Example:
        >>> class MySearchProvider(SearchProvider):
        ...     async def search(self, query: str, limit: int = 10) -> SearchResultBatch:
        ...         # Implementation here
        ...         return SearchResultBatch(results=[...])
        >>> provider = MySearchProvider()
        >>> results = await provider.search("python web scraping")
    """
    
    @property
    def provider_name(self) -> str:
        """Name of this search provider (for logging/CLI).
        
        Returns:
            Provider name string
        """
        return self.__class__.__name__
    
    @abstractmethod
    async def search(
        self,
        query: str,
        limit: int = 10,
        **kwargs: Any,
    ) -> SearchResultBatch:
        """Search the web and return results.
        
        This is the single method that must be overridden by all
        search provider implementations.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            **kwargs: Provider-specific options (e.g., sources, categories)
            
        Returns:
            SearchResultBatch containing search results
            
        Raises:
            ValueError: If query is invalid or search fails
        """
        ...

"""Stub search provider for testing and development.

Returns mock search results without making actual API calls.
Useful for testing the pipeline without requiring API keys.
"""

from spiderweb.search.base import SearchProvider, SearchResult, SearchResultBatch


class StubSearchProvider(SearchProvider):
    """Stub search provider that returns mock results.
    
    Useful for testing and development when you don't have
    a real search API key or want to avoid API costs.
    
    Example:
        >>> provider = StubSearchProvider()
        >>> results = await provider.search("python web scraping", limit=5)
        >>> print(len(results.results))
        5
    """
    
    async def search(
        self,
        query: str,
        limit: int = 10,
        **kwargs: dict,
    ) -> SearchResultBatch:
        """Return mock search results.
        
        Args:
            query: Search query (used to generate mock URLs)
            limit: Number of results to return
            **kwargs: Ignored
            
        Returns:
            SearchResultBatch with mock results
        """
        # Generate mock results based on query
        results = []
        for i in range(limit):
            # Create a mock URL based on query
            url_slug = query.lower().replace(" ", "-")[:30]
            results.append(
                SearchResult(
                    url=f"https://example.com/{url_slug}-{i+1}",
                    title=f"Mock Result {i+1} for '{query}'",
                    description=f"This is a mock search result for the query '{query}'. "
                    f"Result number {i+1}.",
                    position=i + 1,
                    source="web",
                )
            )
        
        return SearchResultBatch(
            results=results,
            query=query,
            total=limit,
        )

"""Tests for Tavily search provider.

Tests initialization, search functionality, and error handling
with mocked Tavily client.
"""

import asyncio
import os
import sys

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from spiderweb.search.tavily import TavilySearchProvider


async def _run_sync(f):
    """Run a sync function and return its result (for patching asyncio.to_thread)."""
    return f()


class TestTavilySearchProviderInit:
    """Tests for TavilySearchProvider initialization."""

    def test_init_with_api_key(self):
        """Initializes successfully with api_key parameter."""
        provider = TavilySearchProvider(api_key="test-api-key")
        assert provider.api_key == "test-api-key"

    def test_init_with_env_var(self):
        """Initializes successfully with TAVILY_API_KEY env var."""
        with patch.dict(os.environ, {"TAVILY_API_KEY": "env-api-key"}):
            provider = TavilySearchProvider()
            assert provider.api_key == "env-api-key"

    def test_init_api_key_overrides_env(self):
        """api_key parameter overrides environment variable."""
        with patch.dict(os.environ, {"TAVILY_API_KEY": "env-key"}):
            provider = TavilySearchProvider(api_key="param-key")
            assert provider.api_key == "param-key"

    def test_init_no_api_key_warning(self):
        """Initializes without API key but logs warning."""
        with patch.dict(os.environ, {}, clear=True):
            provider = TavilySearchProvider()
            assert provider.api_key is None


class TestTavilySearchProviderSearch:
    """Tests for TavilySearchProvider search method."""

    @pytest.mark.asyncio
    async def test_search_with_mocked_client(self):
        """Search returns SearchResultBatch with mocked Tavily client."""
        mock_response = {
            "results": [
                {
                    "url": "https://example.com/page1",
                    "title": "Example Page 1",
                    "content": "This is the content of page 1",
                    "score": 0.95,
                },
                {
                    "url": "https://example.com/page2",
                    "title": "Example Page 2",
                    "content": "This is the content of page 2",
                    "score": 0.85,
                },
            ]
        }

        mock_client = MagicMock()
        mock_client.search = MagicMock(return_value=mock_response)
        mock_tavily = MagicMock()
        mock_tavily.TavilyClient = MagicMock(return_value=mock_client)
        provider = TavilySearchProvider(api_key="test-key")

        with patch.dict(sys.modules, {"tavily": mock_tavily}), \
             patch("spiderweb.search.tavily.asyncio.to_thread", side_effect=_run_sync):
            result_batch = await provider.search("test query", limit=10)

        assert result_batch.query == "test query"
        assert result_batch.total == 2
        assert len(result_batch.results) == 2

        # Check first result
        result1 = result_batch.results[0]
        assert result1.url == "https://example.com/page1"
        assert result1.title == "Example Page 1"
        assert result1.description == "This is the content of page 1"
        assert result1.snippet == "This is the content of page 1"
        assert result1.position == 1
        assert result1.source == "tavily"
        assert result1.metadata["score"] == 0.95

    @pytest.mark.asyncio
    async def test_search_respects_limit(self):
        """Search respects limit parameter."""
        mock_response = {
            "results": [
                {"url": f"https://example.com/page{i}", "title": f"Page {i}", "content": f"Content {i}"}
                for i in range(10)
            ]
        }

        mock_client = MagicMock()
        mock_client.search = MagicMock(return_value=mock_response)
        mock_tavily = MagicMock()
        mock_tavily.TavilyClient = MagicMock(return_value=mock_client)
        provider = TavilySearchProvider(api_key="test-key")

        with patch.dict(sys.modules, {"tavily": mock_tavily}), \
             patch("spiderweb.search.tavily.asyncio.to_thread", side_effect=_run_sync):
            result_batch = await provider.search("test query", limit=5)

        assert len(result_batch.results) == 5
        assert result_batch.total == 5

    @pytest.mark.asyncio
    async def test_search_passes_kwargs(self):
        """Search passes additional kwargs to Tavily client."""
        mock_client = MagicMock()
        mock_response = {"results": []}
        mock_client.search = MagicMock(return_value=mock_response)
        mock_tavily = MagicMock()
        mock_tavily.TavilyClient = MagicMock(return_value=mock_client)
        provider = TavilySearchProvider(api_key="test-key")

        with patch.dict(sys.modules, {"tavily": mock_tavily}), \
             patch("spiderweb.search.tavily.asyncio.to_thread", side_effect=_run_sync):
            await provider.search(
                "test query",
                limit=10,
                search_depth="advanced",
                include_answer=True,
            )

        # Verify search was called with correct kwargs
        call_kwargs = mock_client.search.call_args[1]
        assert call_kwargs["search_depth"] == "advanced"
        assert call_kwargs["include_answer"] is True

    @pytest.mark.asyncio
    async def test_search_no_api_key_raises(self):
        """Search raises ValueError when API key not set."""
        provider = TavilySearchProvider(api_key=None)

        with pytest.raises(ValueError, match="Tavily API key required"):
            await provider.search("test query")

    @pytest.mark.asyncio
    async def test_search_missing_tavily_package(self):
        """Search raises ImportError when tavily-python not installed."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "tavily":
                raise ImportError("No module named 'tavily'")
            return real_import(name, *args, **kwargs)

        provider = TavilySearchProvider(api_key="test-key")

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="tavily-python is not installed"):
                await provider.search("test query")

    @pytest.mark.asyncio
    async def test_search_empty_results(self):
        """Search handles empty results gracefully."""
        mock_response = {"results": []}
        mock_client = MagicMock()
        mock_client.search = MagicMock(return_value=mock_response)
        mock_tavily = MagicMock()
        mock_tavily.TavilyClient = MagicMock(return_value=mock_client)
        provider = TavilySearchProvider(api_key="test-key")

        with patch.dict(sys.modules, {"tavily": mock_tavily}), \
             patch("spiderweb.search.tavily.asyncio.to_thread", side_effect=_run_sync):
            result_batch = await provider.search("test query")

        assert result_batch.total == 0
        assert len(result_batch.results) == 0

    @pytest.mark.asyncio
    async def test_search_missing_fields(self):
        """Search handles missing fields in response gracefully."""
        mock_response = {
            "results": [
                {
                    "url": "https://example.com",
                    # Missing title, content, score
                }
            ]
        }

        mock_client = MagicMock()
        mock_client.search = MagicMock(return_value=mock_response)
        mock_tavily = MagicMock()
        mock_tavily.TavilyClient = MagicMock(return_value=mock_client)
        provider = TavilySearchProvider(api_key="test-key")

        with patch.dict(sys.modules, {"tavily": mock_tavily}), \
             patch("spiderweb.search.tavily.asyncio.to_thread", side_effect=_run_sync):
            result_batch = await provider.search("test query")

        assert len(result_batch.results) == 1
        result = result_batch.results[0]
        assert result.url == "https://example.com"
        assert result.title == ""  # Default empty string
        assert result.description is None  # None when content missing

    @pytest.mark.asyncio
    async def test_search_exception_propagation(self):
        """Search propagates exceptions from Tavily API."""
        async def raise_on_call(f):
            raise Exception("API Error")

        mock_client = MagicMock()
        mock_tavily = MagicMock()
        mock_tavily.TavilyClient = MagicMock(return_value=mock_client)
        provider = TavilySearchProvider(api_key="test-key")

        with patch.dict(sys.modules, {"tavily": mock_tavily}), \
             patch("spiderweb.search.tavily.asyncio.to_thread", side_effect=raise_on_call):
            with pytest.raises(Exception, match="API Error"):
                await provider.search("test query")

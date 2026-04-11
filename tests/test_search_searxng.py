"""Tests for SearXNG search provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spiderweb.registry import search_provider_registry
from spiderweb.search.searxng import SearxNGSearchProvider


def _session_mocks(*, json_payload: dict | None = None, status: int = 200, text: str = "err"):
    mock_resp = MagicMock()
    mock_resp.status = status
    if status != 200:
        mock_resp.text = AsyncMock(return_value=text)
    else:
        mock_resp.json = AsyncMock(return_value=json_payload or {})

    get_cm = MagicMock()
    get_cm.__aenter__ = AsyncMock(return_value=mock_resp)
    get_cm.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=get_cm)

    session_cm = MagicMock()
    session_cm.__aenter__ = AsyncMock(return_value=mock_session)
    session_cm.__aexit__ = AsyncMock(return_value=None)
    return session_cm, mock_session


def test_searxng_registered_in_registry():
    assert "searxng" in search_provider_registry.list()


@pytest.mark.asyncio
async def test_search_maps_results():
    provider = SearxNGSearchProvider(searxng_url="https://sx.test")
    payload = {
        "results": [
            {
                "url": "https://a.example",
                "title": "Title A",
                "content": "Snippet A",
                "engine": "google",
            },
            {"url": "https://b.example", "title": "Title B"},
        ]
    }
    session_cm, mock_session = _session_mocks(json_payload=payload)
    with patch("spiderweb.search.searxng.aiohttp.ClientSession", return_value=session_cm):
        batch = await provider.search("hello", limit=10)

    assert batch.query == "hello"
    assert batch.total == 2
    assert len(batch.results) == 2
    assert batch.results[0].url == "https://a.example"
    assert batch.results[0].title == "Title A"
    assert batch.results[0].description == "Snippet A"
    assert batch.results[0].source == "searxng"
    assert batch.results[0].metadata.get("engine") == "google"

    call_kw = mock_session.get.call_args
    assert call_kw[0][0] == "https://sx.test/search"
    params = call_kw[1]["params"]
    assert params["q"] == "hello"
    assert params["format"] == "json"


@pytest.mark.asyncio
async def test_search_respects_limit():
    provider = SearxNGSearchProvider(searxng_url="https://sx.test")
    payload = {"results": [{"url": f"https://x{i}.test", "title": str(i)} for i in range(20)]}
    session_cm, _ = _session_mocks(json_payload=payload)
    with patch("spiderweb.search.searxng.aiohttp.ClientSession", return_value=session_cm):
        batch = await provider.search("q", limit=3)
    assert len(batch.results) == 3
    assert batch.total == 3


@pytest.mark.asyncio
async def test_search_empty_query_raises():
    provider = SearxNGSearchProvider(searxng_url="https://sx.test")
    with pytest.raises(ValueError, match="empty"):
        await provider.search("   ", limit=5)


@pytest.mark.asyncio
async def test_search_missing_base_url_raises():
    provider = SearxNGSearchProvider(searxng_url="https://sx.test")
    provider._base_url = None
    with pytest.raises(ValueError, match="base URL"):
        await provider.search("q")


@pytest.mark.asyncio
async def test_search_http_error_raises():
    provider = SearxNGSearchProvider(searxng_url="https://sx.test")
    session_cm, _ = _session_mocks(status=503, text="unavailable")
    with (
        patch("spiderweb.search.searxng.aiohttp.ClientSession", return_value=session_cm),
        pytest.raises(ValueError, match="HTTP 503"),
    ):
        await provider.search("q")


@pytest.mark.asyncio
async def test_search_skips_entries_without_url():
    provider = SearxNGSearchProvider(searxng_url="https://sx.test")
    payload = {"results": [{"title": "no url"}, {"url": "https://ok.test", "title": "ok"}]}
    session_cm, _ = _session_mocks(json_payload=payload)
    with patch("spiderweb.search.searxng.aiohttp.ClientSession", return_value=session_cm):
        batch = await provider.search("q", limit=10)
    assert len(batch.results) == 1
    assert batch.results[0].url == "https://ok.test"


@pytest.mark.asyncio
async def test_search_merges_time_range_from_kwargs():
    provider = SearxNGSearchProvider(searxng_url="https://sx.test")
    session_cm, mock_session = _session_mocks(json_payload={"results": []})
    with patch("spiderweb.search.searxng.aiohttp.ClientSession", return_value=session_cm):
        await provider.search("q", limit=5, time_range="month")
    params = mock_session.get.call_args[1]["params"]
    assert params["time_range"] == "month"

"""Tests for URL liveness HEAD checks."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from spiderweb.crawlers.url_validation import validate_url_liveness


def test_validate_url_liveness_404():
    mock_resp = MagicMock()
    mock_resp.status = 404
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_resp)
    cm.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.head = MagicMock(return_value=cm)

    async def run():
        return await validate_url_liveness(
            "https://example.com/missing",
            timeout=2.0,
            session=mock_session,
        )

    ok, status, err = asyncio.run(run())
    assert ok is False
    assert status == 404


def test_validate_url_liveness_200():
    mock_resp = MagicMock()
    mock_resp.status = 200
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_resp)
    cm.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.head = MagicMock(return_value=cm)

    async def run():
        return await validate_url_liveness(
            "https://example.com/",
            timeout=2.0,
            session=mock_session,
        )

    ok, status, err = asyncio.run(run())
    assert ok is True
    assert status == 200


def test_validate_url_liveness_invalid_url():
    async def run():
        return await validate_url_liveness("not-a-url")

    ok, status, err = asyncio.run(run())
    assert ok is False
    assert err

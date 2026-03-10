"""Tests for OpenSky Network crawler."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spiderweb.crawlers.base import CrawlResult
from spiderweb.crawlers.opensky import (
    OpenSkyCrawler,
    OPENSKY_MAX_HISTORY_INTERVAL_HOURS,
    _opensky_historical_window_utc,
    _parse_state_vector,
    QUERY_ICAO24,
    QUERY_DEPARTURES,
    QUERY_ARRIVALS,
    QUERY_AREA,
)
from spiderweb.models.config import CrawlerConfig


class TestQueryParsing:
    """Tests for OpenSky query prefix parsing."""

    def test_icao24_pattern_matches(self):
        assert QUERY_ICAO24.match("icao24:3c6444")
        assert QUERY_ICAO24.match("icao24:abc123")
        assert QUERY_ICAO24.match("icao24:A1B2C3")

    def test_icao24_pattern_rejects_invalid(self):
        assert QUERY_ICAO24.match("icao24:xy") is None
        assert QUERY_ICAO24.match("icao24:1234567") is None
        assert QUERY_ICAO24.match("icao24:ghijkl") is None

    def test_departures_pattern_matches(self):
        assert QUERY_DEPARTURES.match("departures:EDDF")
        assert QUERY_DEPARTURES.match("departures:KJFK")

    def test_arrivals_pattern_matches(self):
        assert QUERY_ARRIVALS.match("arrivals:EDDF")
        assert QUERY_ARRIVALS.match("arrivals:KJFK")

    def test_area_pattern_matches(self):
        m = QUERY_AREA.match("area:45.8,5.9,47.8,10.5")
        assert m is not None
        assert m.group(1) == "45.8"
        assert m.group(2) == "5.9"
        assert m.group(3) == "47.8"
        assert m.group(4) == "10.5"


class TestParseStateVector:
    """Tests for state vector parsing."""

    def test_parses_full_state_vector(self):
        state = [
            "a1b2c3",      # icao24
            "BAW123 ",     # callsign
            "United Kingdom",
            1700000000,     # time_position
            1700000010,     # last_contact
            -0.5,           # longitude
            51.5,           # latitude
            10000.0,        # baro_altitude
            False,          # on_ground
            250.0,          # velocity
            90.0,           # true_track
            5.0,            # vertical_rate
            [1, 2],         # sensors
            10100.0,        # geo_altitude
            "1234",         # squawk
            False,          # spi
            0,              # position_source
            4,              # category
        ]
        parsed = _parse_state_vector(state)
        assert parsed["icao24"] == "a1b2c3"
        assert parsed["callsign"] == "BAW123"
        assert parsed["latitude"] == 51.5
        assert parsed["longitude"] == -0.5
        assert parsed["baro_altitude"] == 10000.0
        assert parsed["velocity"] == 250.0
        assert parsed["on_ground"] is False
        assert parsed["category"] == 4

    def test_handles_short_state_vector(self):
        state = ["abc123", "CALL01", "France"]
        parsed = _parse_state_vector(state)
        assert parsed["icao24"] == "abc123"
        assert parsed["callsign"] == "CALL01"
        assert parsed["latitude"] is None
        assert parsed["last_contact"] is None


class TestOpenskyHistoricalWindow:
    """Regression: history window ends at last completed UTC day, not now."""

    def test_historical_window_ends_at_utc_midnight(self):
        """end_ts must be at UTC midnight (OpenSky only exposes previous day or earlier)."""
        begin_ts, end_ts = _opensky_historical_window_utc(48)
        assert end_ts % 86400 == 0, "end_ts should be at UTC midnight"
        assert begin_ts < end_ts
        assert end_ts - begin_ts == 48 * 3600

    def test_historical_window_caps_lookback_at_48_hours(self):
        """Lookback > 48 is capped per OpenSky API limit."""
        begin_ts, end_ts = _opensky_historical_window_utc(100)
        assert end_ts - begin_ts == OPENSKY_MAX_HISTORY_INTERVAL_HOURS * 3600


class TestOpenSkyCrawlerSearch:
    """Tests for OpenSkyCrawler.search() with mocked API."""

    def test_search_icao24_returns_state_results(self):
        async def _run():
            crawler = OpenSkyCrawler()
            mock_data = {
                "time": 1700000000,
                "states": [
                    [
                        "3c6444", "DLH123", "Germany",
                        1700000000, 1700000010,
                        -0.5, 51.5, 10000.0, False, 250.0, 90.0, 5.0,
                        None, 10100.0, "1234", False, 0, 4,
                    ],
                ],
            }
            with patch.object(crawler, "_request_get", new_callable=AsyncMock, return_value=mock_data):
                results = await crawler.search("icao24:3c6444", config=None)
            await crawler.close()
            return results

        results = asyncio.run(_run())
        assert len(results) >= 1
        cr = results[0]
        assert cr.success is True
        assert cr.metadata["icao24"] == "3c6444"
        assert cr.metadata["callsign"] == "DLH123"
        assert cr.metadata["first_seen"] is not None

    def test_search_unsupported_format_returns_error_result(self):
        async def _run():
            crawler = OpenSkyCrawler()
            results = await crawler.search("unknown:format", config=None)
            await crawler.close()
            return results

        results = asyncio.run(_run())
        assert len(results) == 1
        assert results[0].success is False
        assert "Unsupported" in (results[0].error or "")

    def test_scrape_user_delegates_to_search(self):
        async def _run():
            crawler = OpenSkyCrawler()
            with patch.object(crawler, "search", new_callable=AsyncMock, return_value=[]) as mock_search:
                await crawler.scrape_user("3c6444", config=None)
            await crawler.close()
            return mock_search

        mock_search = asyncio.run(_run())
        mock_search.assert_called_once_with("icao24:3c6444", None)

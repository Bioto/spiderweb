"""OpenSky Network API crawler for flight tracking.

Supports:
- Live state vectors (all aircraft, by ICAO24, or bounding box)
- Flight history by aircraft, departures by airport, arrivals by airport
- search(query) parses prefix syntax: icao24:X, departures:X, arrivals:X, area:lamin,lomin,lamax,lomax
"""

import asyncio
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import aiohttp

from spiderweb.crawlers.base import CrawlResult, Crawler
from spiderweb.models.config import CrawlerConfig, OpenSkyConfig
from spiderweb.observability.logging_config import get_logger

logger = get_logger(__name__)

OPENSKY_API_BASE = "https://opensky-network.org/api"
OPENSKY_MAX_HISTORY_INTERVAL_HOURS = 48


def _opensky_historical_window_utc(
    lookback_hours: int,
) -> tuple[int, int]:
    """Compute (begin_ts, end_ts) for OpenSky historical flight endpoints.

    OpenSky exposes only flights from the previous day or earlier. The end of
    the window must be at most the end of the last completed UTC day.
    The interval must not exceed 2 days (48 hours).

    Returns:
        (begin_ts, end_ts) as Unix timestamps, bounded to a valid API window.
    """
    capped = min(lookback_hours, OPENSKY_MAX_HISTORY_INTERVAL_HOURS)
    now = datetime.now(timezone.utc)
    # End at midnight UTC of the current day (start of today) = end of yesterday
    end_dt = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end_ts = int(end_dt.timestamp())
    begin_ts = end_ts - (capped * 3600)
    return (begin_ts, end_ts)
OPENSKY_TOKEN_URL = (
    "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
)
TOKEN_REFRESH_MARGIN_SECONDS = 30

# State vector field indices from OpenSky REST API
STATE_ICAO24 = 0
STATE_CALLSIGN = 1
STATE_ORIGIN_COUNTRY = 2
STATE_TIME_POSITION = 3
STATE_LAST_CONTACT = 4
STATE_LONGITUDE = 5
STATE_LATITUDE = 6
STATE_BARO_ALTITUDE = 7
STATE_ON_GROUND = 8
STATE_VELOCITY = 9
STATE_TRUE_TRACK = 10
STATE_VERTICAL_RATE = 11
STATE_SENSORS = 12
STATE_GEO_ALTITUDE = 13
STATE_SQUAWK = 14
STATE_SPI = 15
STATE_POSITION_SOURCE = 16
STATE_CATEGORY = 17


def _parse_state_vector(state: list[Any]) -> dict[str, Any]:
    """Convert raw state vector array to dict with named keys."""
    return {
        "icao24": state[STATE_ICAO24] if len(state) > STATE_ICAO24 else None,
        "callsign": (state[STATE_CALLSIGN] or "").strip() if len(state) > STATE_CALLSIGN else None,
        "origin_country": state[STATE_ORIGIN_COUNTRY] if len(state) > STATE_ORIGIN_COUNTRY else None,
        "time_position": state[STATE_TIME_POSITION] if len(state) > STATE_TIME_POSITION else None,
        "last_contact": state[STATE_LAST_CONTACT] if len(state) > STATE_LAST_CONTACT else None,
        "longitude": state[STATE_LONGITUDE] if len(state) > STATE_LONGITUDE else None,
        "latitude": state[STATE_LATITUDE] if len(state) > STATE_LATITUDE else None,
        "baro_altitude": state[STATE_BARO_ALTITUDE] if len(state) > STATE_BARO_ALTITUDE else None,
        "on_ground": state[STATE_ON_GROUND] if len(state) > STATE_ON_GROUND else False,
        "velocity": state[STATE_VELOCITY] if len(state) > STATE_VELOCITY else None,
        "true_track": state[STATE_TRUE_TRACK] if len(state) > STATE_TRUE_TRACK else None,
        "vertical_rate": state[STATE_VERTICAL_RATE] if len(state) > STATE_VERTICAL_RATE else None,
        "geo_altitude": state[STATE_GEO_ALTITUDE] if len(state) > STATE_GEO_ALTITUDE else None,
        "squawk": state[STATE_SQUAWK] if len(state) > STATE_SQUAWK else None,
        "category": state[STATE_CATEGORY] if len(state) > STATE_CATEGORY else None,
    }


# Query prefix patterns: icao24:X, departures:X, arrivals:X, area:lamin,lomin,lamax,lomax
QUERY_ICAO24 = re.compile(r"^icao24:([a-fA-F0-9]{6})$")
QUERY_DEPARTURES = re.compile(r"^departures:([A-Za-z0-9]{4})$")
QUERY_ARRIVALS = re.compile(r"^arrivals:([A-Za-z0-9]{4})$")
QUERY_AREA = re.compile(r"^area:([-\d.]+),([-\d.]+),([-\d.]+),([-\d.]+)$")


def _get_opensky_config(config: CrawlerConfig | None) -> OpenSkyConfig:
    """Resolve OpenSkyConfig from CrawlerConfig.extra_config or default."""
    if config and config.extra_config.get("opensky_config"):
        raw = config.extra_config["opensky_config"]
        return (
            OpenSkyConfig.model_validate(raw)
            if isinstance(raw, dict)
            else raw
        )
    return OpenSkyConfig()


def _get_credentials(config: CrawlerConfig | None) -> tuple[str | None, str | None]:
    """Get (client_id, client_secret) from config or settings."""
    if config and config.extra_config:
        cid = config.extra_config.get("opensky_client_id")
        csec = config.extra_config.get("opensky_client_secret")
        if cid is not None and csec is not None:
            return (cid, csec)
    try:
        from spiderweb.config import settings

        return (
            getattr(settings, "opensky_client_id", None),
            getattr(settings, "opensky_client_secret", None),
        )
    except Exception:
        return (None, None)


class OpenSkyTokenManager:
    """Manages OAuth2 client credentials tokens for OpenSky API."""

    def __init__(self, client_id: str, client_secret: str) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._token: str | None = None
        self._expires_at: datetime | None = None

    def is_valid(self) -> bool:
        """Return True if we have a non-expired token."""
        if not self._token or not self._expires_at:
            return False
        return datetime.now(timezone.utc) < self._expires_at

    async def get_token(self, session: aiohttp.ClientSession) -> str:
        """Return a valid access token, refreshing if needed."""
        if self.is_valid():
            return self._token  # type: ignore
        return await self._refresh(session)

    async def _refresh(self, session: aiohttp.ClientSession) -> str:
        """Fetch a new access token."""
        data = {
            "grant_type": "client_credentials",
            "client_id": self._client_id,
            "client_secret": self._client_secret,
        }
        logger.debug(
            "Requesting OpenSky token for client_id=%s...",
            self._client_id[:8] if self._client_id else "NONE",
        )
        async with session.post(OPENSKY_TOKEN_URL, data=data) as resp:
            if resp.status != 200:
                body = await resp.text()
                logger.error(
                    "OpenSky token request failed: status=%d body=%s",
                    resp.status,
                    body[:500],
                )
                resp.raise_for_status()
            payload = await resp.json()
        self._token = payload["access_token"]
        expires_in = payload.get("expires_in", 1800)
        self._expires_at = datetime.now(timezone.utc) + timedelta(
            seconds=expires_in - TOKEN_REFRESH_MARGIN_SECONDS
        )
        return self._token

    def headers(self, session: aiohttp.ClientSession) -> dict[str, str]:
        """Return headers with Authorization Bearer (caller must await get_token first)."""
        return {"Authorization": f"Bearer {self._token}"}


class OpenSkyCrawler:
    """Crawler for OpenSky Network flight tracking API.

    - search(query): parses prefix query format, dispatches to states/flights API
    - get_states(): live state vectors
    - get_flights_by_aircraft(), get_departures(), get_arrivals(): flight history
    - crawl() / crawl_many(): Crawler protocol stubs (search is primary for this crawler)
    """

    def __init__(self) -> None:
        self._session: aiohttp.ClientSession | None = None
        self._token_manager: OpenSkyTokenManager | None = None
        logger.debug("Initialized OpenSkyCrawler")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _request_get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        config: CrawlerConfig | None = None,
        timeout_seconds: int = 30,
    ) -> dict[str, Any] | list[Any]:
        """GET a path on OpenSky API. Returns JSON."""
        session = await self._get_session()
        url = f"{OPENSKY_API_BASE.rstrip('/')}/{path.lstrip('/')}"
        headers: dict[str, str] = {}

        if self._token_manager:
            token = await self._token_manager.get_token(session)
            headers["Authorization"] = f"Bearer {token}"

        async with session.get(
            url,
            params=params or {},
            headers=headers or None,
            timeout=aiohttp.ClientTimeout(total=timeout_seconds),
        ) as resp:
            if resp.status == 404:
                return [] if "flights" in path else {"time": 0, "states": None}
            resp.raise_for_status()
            return await resp.json()

    async def get_states(
        self,
        icao24s: list[str] | None = None,
        bounding_box: tuple[float, float, float, float] | None = None,
        time_secs: int | None = None,
        config: CrawlerConfig | None = None,
    ) -> list[dict[str, Any]]:
        """Get state vectors. Optional: filter by ICAO24, bounding box, or time."""
        params: dict[str, Any] = {}
        if time_secs is not None:
            params["time"] = time_secs
        if icao24s:
            # OpenSky API expects repeated icao24 param; aiohttp accepts list
            params["icao24"] = [a.lower() for a in icao24s]
        if bounding_box:
            params["lamin"], params["lomin"], params["lamax"], params["lomax"] = bounding_box

        data = await self._request_get("states/all", params or None, config=config)

        if isinstance(data, list):
            return []
        states_raw = data.get("states") or []
        return [_parse_state_vector(s) for s in states_raw if isinstance(s, (list, tuple))]

    async def get_flights_by_aircraft(
        self,
        icao24: str,
        begin: int,
        end: int,
        config: CrawlerConfig | None = None,
    ) -> list[dict[str, Any]]:
        """Get flight history for an aircraft. begin/end are Unix timestamps."""
        data = await self._request_get(
            "flights/aircraft",
            {"icao24": icao24.lower(), "begin": begin, "end": end},
            config=config,
        )
        return data if isinstance(data, list) else []

    async def get_departures(
        self,
        airport: str,
        begin: int,
        end: int,
        config: CrawlerConfig | None = None,
    ) -> list[dict[str, Any]]:
        """Get departures from an airport. begin/end are Unix timestamps."""
        data = await self._request_get(
            "flights/departure",
            {"airport": airport.upper(), "begin": begin, "end": end},
            config=config,
        )
        return data if isinstance(data, list) else []

    async def get_arrivals(
        self,
        airport: str,
        begin: int,
        end: int,
        config: CrawlerConfig | None = None,
    ) -> list[dict[str, Any]]:
        """Get arrivals at an airport. begin/end are Unix timestamps."""
        data = await self._request_get(
            "flights/arrival",
            {"airport": airport.upper(), "begin": begin, "end": end},
            config=config,
        )
        return data if isinstance(data, list) else []

    def _state_to_crawl_result(
        self, state: dict[str, Any], api_time: int
    ) -> CrawlResult:
        """Convert a state vector dict to CrawlResult."""
        icao24 = state.get("icao24") or "unknown"
        callsign = (state.get("callsign") or "").strip() or icao24
        lat = state.get("latitude")
        lon = state.get("longitude")
        alt = state.get("baro_altitude")
        vel = state.get("velocity")
        track = state.get("true_track")
        url = f"https://opensky-network.org/aircraft-profile/{icao24}"

        lines = [
            f"# Flight {callsign} ({icao24})",
            "",
            f"Origin: {state.get('origin_country', 'Unknown')}",
            f"Position: {lat}, {lon}" if lat is not None and lon is not None else "Position: Unknown",
            f"Altitude: {alt:.0f}m" if alt is not None else "Altitude: Unknown",
            f"Speed: {vel:.0f} m/s" if vel is not None else "Speed: Unknown",
            f"Heading: {track:.0f}°" if track is not None else "",
            f"On ground: {state.get('on_ground', False)}",
            "",
            f"Source: {url}",
        ]
        content = "\n".join(l for l in lines if l)

        return CrawlResult(
            url=url,
            content=content,
            markdown=content,
            status_code=200,
            metadata={
                "source_type": "opensky_state",
                "icao24": icao24,
                "callsign": callsign,
                "first_seen": state.get("time_position") or state.get("last_contact") or api_time,
                "last_seen": state.get("last_contact") or api_time,
                **{k: v for k, v in state.items()},
            },
            links=[],
            success=True,
            error=None,
        )

    def _flight_to_crawl_result(self, flight: dict[str, Any]) -> CrawlResult:
        """Convert a flight object (from departures/arrivals/aircraft) to CrawlResult."""
        icao24 = (flight.get("icao24") or "unknown").lower()
        callsign = (flight.get("callsign") or "").strip() or icao24
        first_seen = flight.get("firstSeen") or flight.get("lastSeen") or int(time.time())
        departure = flight.get("estDepartureAirport") or "?"
        arrival = flight.get("estArrivalAirport") or "?"
        url = f"https://opensky-network.org/aircraft-profile/{icao24}"

        lines = [
            f"# Flight {callsign} ({icao24})",
            "",
            f"Route: {departure} → {arrival}",
            f"First seen: {first_seen}",
            f"Last seen: {flight.get('lastSeen', '?')}",
            "",
            f"Source: {url}",
        ]
        content = "\n".join(lines)

        return CrawlResult(
            url=url,
            content=content,
            markdown=content,
            status_code=200,
            metadata={
                "source_type": "opensky_flight",
                "icao24": icao24,
                "callsign": callsign,
                "first_seen": first_seen,
                "last_seen": flight.get("lastSeen"),
                "departure_airport": departure,
                "arrival_airport": arrival,
                "raw_flight": flight,
            },
            links=[],
            success=True,
            error=None,
        )

    async def search(
        self,
        query: str,
        config: CrawlerConfig | None = None,
    ) -> list[CrawlResult]:
        """Parse prefix query format and dispatch to the appropriate API.

        Supported formats:
        - icao24:3c6444 - track specific aircraft (live states + flight history)
        - departures:EDDF - departures from airport
        - arrivals:KJFK - arrivals at airport
        - area:45.8,5.9,47.8,10.5 - aircraft in bounding box (lamin,lomin,lamax,lomax)
        """
        if config is None:
            config = CrawlerConfig(provider="opensky")
        opensky_config = _get_opensky_config(config)
        client_id, client_secret = _get_credentials(config)
        if client_id and client_secret and self._token_manager is None:
            self._token_manager = OpenSkyTokenManager(client_id, client_secret)

        q = query.strip()

        # icao24:xxx - live states + flight history
        m = QUERY_ICAO24.match(q)
        if m:
            icao24 = m.group(1).lower()
            results: list[CrawlResult] = []

            # Live states
            await asyncio.sleep(opensky_config.delay_between_requests)
            data = await self._request_get(
                "states/all",
                {"icao24": icao24},
                config=config,
            )
            if isinstance(data, dict):
                api_time = data.get("time", int(time.time()))
                states_raw = data.get("states") or []
                for s in states_raw:
                    if isinstance(s, (list, tuple)):
                        parsed = _parse_state_vector(s)
                        results.append(self._state_to_crawl_result(parsed, api_time))

            # Flight history (OpenSky exposes only previous day or earlier)
            if opensky_config.fetch_flight_history:
                begin_ts, end_ts = _opensky_historical_window_utc(
                    opensky_config.history_lookback_hours
                )
                await asyncio.sleep(opensky_config.delay_between_requests)
                flights = await self.get_flights_by_aircraft(icao24, begin_ts, end_ts, config)
                for f in flights:
                    results.append(self._flight_to_crawl_result(f))

            return results

        # departures:XXXX (OpenSky exposes only previous day or earlier)
        m = QUERY_DEPARTURES.match(q)
        if m:
            airport = m.group(1).upper()
            lookback = min(
                opensky_config.departures_arrivals_lookback_hours,
                OPENSKY_MAX_HISTORY_INTERVAL_HOURS,
            )
            begin_ts, end_ts = _opensky_historical_window_utc(lookback)
            await asyncio.sleep(opensky_config.delay_between_requests)
            flights = await self.get_departures(airport, begin_ts, end_ts, config)
            return [self._flight_to_crawl_result(f) for f in flights]

        # arrivals:XXXX (OpenSky exposes only previous day or earlier)
        m = QUERY_ARRIVALS.match(q)
        if m:
            airport = m.group(1).upper()
            lookback = min(
                opensky_config.departures_arrivals_lookback_hours,
                OPENSKY_MAX_HISTORY_INTERVAL_HOURS,
            )
            begin_ts, end_ts = _opensky_historical_window_utc(lookback)
            await asyncio.sleep(opensky_config.delay_between_requests)
            flights = await self.get_arrivals(airport, begin_ts, end_ts, config)
            return [self._flight_to_crawl_result(f) for f in flights]

        # area:lamin,lomin,lamax,lomax
        m = QUERY_AREA.match(q)
        if m:
            lamin, lomin, lamax, lomax = float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))
            await asyncio.sleep(opensky_config.delay_between_requests)
            data = await self._request_get(
                "states/all",
                {"lamin": lamin, "lomin": lomin, "lamax": lamax, "lomax": lomax},
                config=config,
            )
            if isinstance(data, dict):
                api_time = data.get("time", int(time.time()))
                states_raw = data.get("states") or []
                return [
                    self._state_to_crawl_result(_parse_state_vector(s), api_time)
                    for s in states_raw
                    if isinstance(s, (list, tuple))
                ]
            return []

        # Unknown format
        return [
            CrawlResult(
                url="",
                content="",
                status_code=0,
                success=False,
                error=f"Unsupported OpenSky query format: {q!r}. Use icao24:X, departures:X, arrivals:X, or area:lamin,lomin,lamax,lomax",
            )
        ]

    async def scrape_user(
        self,
        _handle: str,
        config: CrawlerConfig | None = None,
    ) -> list[CrawlResult]:
        """OpenSky has no user concept; treat handle as icao24 and delegate to search."""
        return await self.search(f"icao24:{_handle}", config)

    async def crawl(self, url: str, config: CrawlerConfig | None = None) -> CrawlResult:
        """Crawl protocol: OpenSky uses search(); URLs like opensky://icao24:abc123 are handled."""
        if url.startswith("opensky://") or "opensky" in url.lower():
            q = url.replace("opensky://", "").strip()
            results = await self.search(q, config)
            return results[0] if results else CrawlResult(
                url=url, content="", status_code=0, success=False, error="No results"
            )
        return CrawlResult(
            url=url,
            content="",
            status_code=0,
            success=False,
            error="OpenSky crawler expects search queries (icao24:X, departures:X, etc.), not arbitrary URLs",
        )

    async def crawl_many(
        self,
        urls: list[str],
        config: CrawlerConfig | None = None,
    ) -> list[CrawlResult]:
        """Crawl protocol: run search for each OpenSky-style query."""
        results: list[CrawlResult] = []
        for url in urls:
            r = await self.crawl(url, config)
            results.append(r)
        return results

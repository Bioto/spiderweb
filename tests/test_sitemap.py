"""Tests for sitemap parsing functionality.

Tests URL discovery, XML parsing, namespace handling, and sitemap index
following with mocked HTTP requests.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from spiderweb.crawlers.sitemap import (
    discover_sitemap_url,
    parse_sitemap,
    _parse_urlset,
    _get_namespace,
)


class TestDiscoverSitemapUrl:
    """Tests for discover_sitemap_url function."""

    def test_discover_sitemap_url_from_base(self):
        """Returns sitemap.xml URL from base URL."""
        url = discover_sitemap_url("https://example.com")
        assert url == "https://example.com/sitemap.xml"

    def test_discover_sitemap_url_from_path(self):
        """Returns sitemap.xml URL even when base URL has path."""
        url = discover_sitemap_url("https://example.com/page")
        assert url == "https://example.com/sitemap.xml"

    def test_discover_sitemap_url_preserves_scheme(self):
        """Preserves HTTP scheme."""
        url = discover_sitemap_url("http://example.com")
        assert url == "http://example.com/sitemap.xml"


class TestParseUrlset:
    """Tests for _parse_urlset function."""

    def test_parse_urlset_simple(self):
        """Parses simple sitemap with URLs."""
        import xml.etree.ElementTree as ET

        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url>
                <loc>https://example.com/page1</loc>
            </url>
            <url>
                <loc>https://example.com/page2</loc>
            </url>
        </urlset>"""
        root = ET.fromstring(xml_content)
        
        urls = _parse_urlset(root)
        assert len(urls) == 2
        assert "https://example.com/page1" in urls
        assert "https://example.com/page2" in urls

    def test_parse_urlset_no_namespace(self):
        """Parses sitemap without namespace."""
        import xml.etree.ElementTree as ET

        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
        <urlset>
            <url>
                <loc>https://example.com/page1</loc>
            </url>
        </urlset>"""
        root = ET.fromstring(xml_content)
        
        urls = _parse_urlset(root)
        assert len(urls) == 1
        assert urls[0] == "https://example.com/page1"

    def test_parse_urlset_empty(self):
        """Empty sitemap returns empty list."""
        import xml.etree.ElementTree as ET

        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
        </urlset>"""
        root = ET.fromstring(xml_content)
        
        urls = _parse_urlset(root)
        assert urls == []


class TestGetNamespace:
    """Tests for _get_namespace function."""

    def test_get_namespace_standard(self):
        """Extracts standard sitemap namespace."""
        import xml.etree.ElementTree as ET

        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
        </urlset>"""
        root = ET.fromstring(xml_content)
        
        namespace = _get_namespace(root)
        assert namespace == "{http://www.sitemaps.org/schemas/sitemap/0.9}"

    def test_get_namespace_no_namespace(self):
        """Returns empty string for no namespace."""
        import xml.etree.ElementTree as ET

        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
        <urlset>
        </urlset>"""
        root = ET.fromstring(xml_content)
        
        namespace = _get_namespace(root)
        assert namespace == ""

    def test_get_namespace_custom(self):
        """Extracts custom namespace."""
        import xml.etree.ElementTree as ET

        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://example.com/custom">
        </urlset>"""
        root = ET.fromstring(xml_content)
        
        namespace = _get_namespace(root)
        assert namespace == "{http://example.com/custom}"


class TestParseSitemap:
    """Tests for parse_sitemap function with mocked HTTP."""

    @pytest.mark.asyncio
    async def test_parse_sitemap_success(self):
        """Parses sitemap successfully with mocked HTTP."""
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url>
                <loc>https://example.com/page1</loc>
            </url>
            <url>
                <loc>https://example.com/page2</loc>
            </url>
        </urlset>"""

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value=xml_content)

        # session.get() must return an async context manager (object with __aenter__/__aexit__)
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_cm)
        mock_session.close = AsyncMock()

        with patch("spiderweb.crawlers.sitemap.aiohttp.ClientSession", return_value=mock_session):
            urls = await parse_sitemap("https://example.com/sitemap.xml")

        assert len(urls) == 2
        assert "https://example.com/page1" in urls
        assert "https://example.com/page2" in urls

    @pytest.mark.asyncio
    async def test_parse_sitemap_404(self):
        """Returns empty list for 404 response."""
        mock_response = AsyncMock()
        mock_response.status = 404

        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_cm)
        mock_session.close = AsyncMock()

        with patch("spiderweb.crawlers.sitemap.aiohttp.ClientSession", return_value=mock_session):
            urls = await parse_sitemap("https://example.com/sitemap.xml")

        assert urls == []

    @pytest.mark.asyncio
    async def test_parse_sitemap_500(self):
        """Returns empty list for 500 response."""
        mock_response = AsyncMock()
        mock_response.status = 500

        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_cm)
        mock_session.close = AsyncMock()

        with patch("spiderweb.crawlers.sitemap.aiohttp.ClientSession", return_value=mock_session):
            urls = await parse_sitemap("https://example.com/sitemap.xml")

        assert urls == []

    @pytest.mark.asyncio
    async def test_parse_sitemap_index(self):
        """Follows sitemap index to nested sitemaps."""
        index_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <sitemap>
                <loc>https://example.com/sitemap1.xml</loc>
            </sitemap>
            <sitemap>
                <loc>https://example.com/sitemap2.xml</loc>
            </sitemap>
        </sitemapindex>"""

        sitemap1_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url>
                <loc>https://example.com/page1</loc>
            </url>
        </urlset>"""

        sitemap2_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url>
                <loc>https://example.com/page2</loc>
            </url>
        </urlset>"""

        call_count = 0

        async def mock_get(url, **kwargs):
            nonlocal call_count
            mock_response = AsyncMock()
            call_count += 1
            
            if "sitemap_index.xml" in url or call_count == 1:
                mock_response.status = 200
                mock_response.text = AsyncMock(return_value=index_xml)
            elif "sitemap1.xml" in url:
                mock_response.status = 200
                mock_response.text = AsyncMock(return_value=sitemap1_xml)
            elif "sitemap2.xml" in url:
                mock_response.status = 200
                mock_response.text = AsyncMock(return_value=sitemap2_xml)
            else:
                mock_response.status = 404
                mock_response.text = AsyncMock(return_value="")
            
            return mock_response

        mock_session = AsyncMock()
        mock_session.get = AsyncMock(side_effect=mock_get)
        mock_session.get.return_value.__aenter__ = lambda self: self
        mock_session.get.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_session.close = AsyncMock()

        # Patch to return our mock session
        with patch("spiderweb.crawlers.sitemap.aiohttp.ClientSession", return_value=mock_session):
            # Need to patch the actual get call inside _parse_sitemap_recursive
            # This is a bit tricky, so we'll use a simpler approach
            from spiderweb.crawlers.sitemap import _parse_sitemap_recursive
            
            async def mock_get_context(url, **kwargs):
                return await mock_get(url, **kwargs)
            
            mock_session.get = AsyncMock()
            mock_session.get.return_value.__aenter__ = lambda: mock_get_context("", **{})
            mock_session.get.return_value.__aexit__ = AsyncMock(return_value=None)
            
            # Actually, let's use a simpler approach - mock the response objects directly
            responses = {
                "sitemap_index.xml": (200, index_xml),
                "sitemap1.xml": (200, sitemap1_xml),
                "sitemap2.xml": (200, sitemap2_xml),
            }
            
            async def create_mock_response(url):
                for key, (status, content) in responses.items():
                    if key in url:
                        mock_resp = AsyncMock()
                        mock_resp.status = status
                        mock_resp.text = AsyncMock(return_value=content)
                        return mock_resp
                mock_resp = AsyncMock()
                mock_resp.status = 404
                mock_resp.text = AsyncMock(return_value="")
                return mock_resp
            
            class MockContextManager:
                def __init__(self, url):
                    self.url = url
                    self.response = None
                
                async def __aenter__(self):
                    self.response = await create_mock_response(self.url)
                    return self.response
                
                async def __aexit__(self, *args):
                    pass
            
            mock_session.get = lambda url, **kwargs: MockContextManager(url)
            
            urls = await parse_sitemap("https://example.com/sitemap_index.xml", session=mock_session)

        assert len(urls) == 2
        assert "https://example.com/page1" in urls
        assert "https://example.com/page2" in urls

    @pytest.mark.asyncio
    async def test_parse_sitemap_cycle_prevention(self):
        """Prevents infinite loops from circular sitemap references."""
        index_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <sitemap>
                <loc>https://example.com/sitemap_index.xml</loc>
            </sitemap>
        </sitemapindex>"""

        call_count = 0

        async def create_mock_response(url):
            nonlocal call_count
            call_count += 1
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.text = AsyncMock(return_value=index_xml)
            return mock_resp

        class MockContextManager:
            def __init__(self, url):
                self.url = url
                self.response = None
            
            async def __aenter__(self):
                self.response = await create_mock_response(self.url)
                return self.response
            
            async def __aexit__(self, *args):
                pass

        mock_session = AsyncMock()
        mock_session.get = lambda url, **kwargs: MockContextManager(url)
        mock_session.close = AsyncMock()

        urls = await parse_sitemap("https://example.com/sitemap_index.xml", session=mock_session)

        # Should return empty list (cycle detected, no URLs found)
        assert urls == []

    @pytest.mark.asyncio
    async def test_parse_sitemap_with_existing_session(self):
        """Uses provided session instead of creating new one."""
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url>
                <loc>https://example.com/page1</loc>
            </url>
        </urlset>"""

        async def create_mock_response(url):
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.text = AsyncMock(return_value=xml_content)
            return mock_resp

        class MockContextManager:
            def __init__(self, url):
                self.url = url
                self.response = None
            
            async def __aenter__(self):
                self.response = await create_mock_response(self.url)
                return self.response
            
            async def __aexit__(self, *args):
                pass

        mock_session = AsyncMock()
        mock_session.get = lambda url, **kwargs: MockContextManager(url)
        mock_session.close = AsyncMock()

        urls = await parse_sitemap("https://example.com/sitemap.xml", session=mock_session)

        assert len(urls) == 1
        # Should not close the provided session
        mock_session.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_parse_sitemap_exception_handling(self):
        """Handles exceptions gracefully."""
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(side_effect=Exception("Network error"))
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_cm)
        mock_session.close = AsyncMock()

        urls = await parse_sitemap("https://example.com/sitemap.xml", session=mock_session)

        assert urls == []

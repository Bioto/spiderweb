"""Tests for web crawler components."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from spiderweb.crawlers.base import CrawlResult
from spiderweb.crawlers.http import HttpCrawler
from spiderweb.crawlers.extraction import CrawlExtractor
from spiderweb.crawlers.x import (
    XCrawler,
    XUserScrapeResult,
    _get_x_scraper_config,
    _is_x_status_url,
    _tweet_id_from_url,
)
from spiderweb.models.config import XScraperConfig
from spiderweb.loaders.web_loader import WebLoader
from spiderweb.models.config import CrawlerConfig, CrawlExtractionConfig
from pydantic import BaseModel


class TestCrawlResult:
    """Tests for CrawlResult dataclass."""
    
    def test_crawl_result_success(self):
        """Test successful crawl result."""
        result = CrawlResult(
            url="https://example.com",
            content="<html>Hello</html>",
            markdown="Hello",
            status_code=200,
            success=True,
        )
        
        assert result.url == "https://example.com"
        assert result.success is True
        assert result.error is None
    
    def test_crawl_result_failure(self):
        """Test failed crawl result."""
        result = CrawlResult(
            url="https://example.com",
            content="",
            status_code=404,
            success=False,
            error="Not found",
        )
        
        assert result.success is False
        assert result.error == "Not found"
    
    def test_crawl_result_auto_fail_on_error(self):
        """Test that error automatically sets success to False."""
        result = CrawlResult(
            url="https://example.com",
            content="",
            status_code=500,
            success=True,  # Should be overridden
            error="Server error",
        )
        
        assert result.success is False


class TestHttpCrawler:
    """Tests for HttpCrawler."""
    
    async def test_http_crawler_init(self):
        """Test HttpCrawler initialization."""
        crawler = HttpCrawler()
        assert crawler._session is None
        await crawler.close()
    
    async def test_extract_links(self):
        """Test link extraction from HTML."""
        crawler = HttpCrawler()
        
        html = """
        <html>
            <a href="https://example.com/page1">Page 1</a>
            <a href="/page2">Page 2</a>
            <a href="#anchor">Anchor</a>
            <a href="javascript:void(0)">JavaScript</a>
        </html>
        """
        
        links = crawler._extract_links(html, "https://example.com")
        
        # Should extract absolute URLs and resolve relative ones
        assert "https://example.com/page1" in links
        assert "https://example.com/page2" in links
        # Should not include anchors or javascript links
        assert not any("#" in link for link in links)
        assert not any("javascript:" in link for link in links)
        
        await crawler.close()
    
    async def test_should_follow_link(self):
        """Test link following logic."""
        crawler = HttpCrawler()
        
        config = CrawlerConfig(
            provider="http",
            follow_patterns=[r"example\.com/docs"],
            exclude_patterns=[r"example\.com/admin"],
        )
        
        # Should follow matching pattern
        assert crawler._should_follow_link("https://example.com/docs/page", config)
        
        # Should not follow excluded pattern
        assert not crawler._should_follow_link("https://example.com/admin/page", config)
        
        # Should not follow if doesn't match follow patterns
        assert not crawler._should_follow_link("https://example.com/blog/page", config)
        
        await crawler.close()


class TestCrawlExtractor:
    """Tests for CrawlExtractor."""
    
    async def test_crawl_extractor_init(self):
        """Test CrawlExtractor initialization."""
        llm_client = MagicMock()
        config = CrawlExtractionConfig(enabled=True)
        
        extractor = CrawlExtractor(llm_client, config)
        
        assert extractor.llm_client == llm_client
        assert extractor.config == config
    
    def test_get_schema_description(self):
        """Test schema description generation."""
        class Product(BaseModel):
            """A product from an online store."""
            name: str
            price: float
            description: str = ""
        
        llm_client = MagicMock()
        extractor = CrawlExtractor(llm_client)
        
        description = extractor._get_schema_description(Product)
        
        assert "Product" in description
        assert "name" in description
        assert "price" in description
        assert "description" in description
    
    def test_build_extraction_prompt(self):
        """Test extraction prompt building."""
        class Product(BaseModel):
            name: str
            price: float
        
        llm_client = MagicMock()
        extractor = CrawlExtractor(llm_client)
        
        content = "<html><h1>Product Name</h1><p>Price: $19.99</p></html>"
        
        prompt = extractor._build_extraction_prompt(
            content,
            schema=Product,
            semantic_guide="Extract product information",
            extraction_query="Get the product name and price",
        )
        
        assert "Extract product information" in prompt
        assert "Get the product name and price" in prompt
        assert "Product" in prompt
        assert content in prompt
    
    def test_parse_llm_response_json(self):
        """Test parsing LLM JSON response."""
        class Product(BaseModel):
            name: str
            price: float
        
        llm_client = MagicMock()
        extractor = CrawlExtractor(llm_client)
        
        # Test plain JSON
        response = '{"name": "Widget", "price": 19.99}'
        result = extractor._parse_llm_response(response, schema=Product)
        
        assert isinstance(result, Product)
        assert result.name == "Widget"
        assert result.price == 19.99
    
    def test_parse_llm_response_markdown(self):
        """Test parsing LLM response with markdown code blocks."""
        class Product(BaseModel):
            name: str
            price: float
        
        llm_client = MagicMock()
        extractor = CrawlExtractor(llm_client)
        
        # Test JSON wrapped in markdown
        response = '```json\n{"name": "Widget", "price": 19.99}\n```'
        result = extractor._parse_llm_response(response, schema=Product)
        
        assert isinstance(result, Product)
        assert result.name == "Widget"
    
    async def test_extract_with_schema(self):
        """Test extraction with Pydantic schema."""
        class Product(BaseModel):
            name: str
            price: float
        
        # Mock LLM client
        llm_client = AsyncMock()
        llm_client.generate = AsyncMock(return_value=MagicMock(
            text='{"name": "Widget", "price": 19.99}'
        ))
        
        extractor = CrawlExtractor(llm_client)
        
        content = "<html><h1>Widget</h1><p>$19.99</p></html>"
        result = await extractor.extract(
            content,
            schema=Product,
            semantic_guide="Extract product information",
        )
        
        assert isinstance(result, Product)
        assert result.name == "Widget"
        assert result.price == 19.99
        
        # Verify LLM was called
        llm_client.generate.assert_called_once()


class TestWebLoader:
    """Tests for WebLoader."""
    
    async def test_web_loader_init(self):
        """Test WebLoader initialization."""
        loader = WebLoader()
        
        assert loader.crawler is not None
        assert loader.crawler_config is not None
        assert loader.extraction_config is not None
    
    async def test_web_loader_with_http_provider(self):
        """Test WebLoader with HTTP crawler."""
        config = CrawlerConfig(provider="http")
        loader = WebLoader(crawler_config=config)
        
        assert isinstance(loader.crawler, HttpCrawler)
    
    async def test_crawl_result_to_document(self):
        """Test converting CrawlResult to Document."""
        loader = WebLoader()
        
        result = CrawlResult(
            url="https://example.com",
            content="<html>Hello</html>",
            markdown="# Hello",
            status_code=200,
            metadata={"test": "value"},
            success=True,
        )
        
        document = loader._crawl_result_to_document(result)
        
        assert document.markdown_content == "# Hello"
        assert document.metadata.source == "https://example.com"
        assert document.metadata.file_type == "html"
        assert document.metadata.extra["status_code"] == 200
        assert document.metadata.extra["test"] == "value"


class TestXCrawler:
    """Tests for X (Twitter) API crawler."""

    def test_is_x_status_url(self):
        """X status URLs are detected."""
        assert _is_x_status_url("https://x.com/user/status/1234567890") is True
        assert _is_x_status_url("https://twitter.com/user/status/1234567890") is True
        assert _is_x_status_url("https://www.x.com/handle/status/999") is True
        assert _is_x_status_url("https://example.com") is False
        assert _is_x_status_url("https://x.com/user") is False

    def test_tweet_id_from_url(self):
        """Tweet ID is extracted from status URL."""
        assert _tweet_id_from_url("https://x.com/foo/status/1346889436626259968") == "1346889436626259968"
        assert _tweet_id_from_url("https://twitter.com/bar/status/123/") == "123"
        assert _tweet_id_from_url("https://example.com") is None

    async def test_x_crawler_non_x_url_returns_error(self):
        """Non-X URL returns failed CrawlResult with clear error."""
        crawler = XCrawler()
        result = await crawler.crawl("https://example.com/page", CrawlerConfig(provider="x"))
        assert result.success is False
        assert "not an X/Twitter status URL" in (result.error or "")
        await crawler.close()

    async def test_x_crawler_missing_token_returns_error(self):
        """Missing Bearer token returns failed CrawlResult."""
        crawler = XCrawler()
        with patch("spiderweb.crawlers.x.settings") as mock_settings:
            mock_settings.x_bearer_token = None
            result = await crawler.crawl(
                "https://x.com/user/status/123",
                CrawlerConfig(provider="x", extra_config={}),
            )
        assert result.success is False
        assert "Bearer token" in (result.error or "")
        await crawler.close()

    async def test_x_crawler_success_mocked(self):
        """X crawler returns content when API returns tweet data."""
        crawler = XCrawler()
        payload = {
            "data": {
                "id": "1346889436626259968",
                "text": "Hello from the API",
                "author_id": "2244994945",
                "created_at": "2021-01-01T12:00:00.000Z",
            },
            "includes": {
                "users": [
                    {"id": "2244994945", "username": "testuser", "name": "Test User"}
                ]
            },
        }
        config = CrawlerConfig(provider="x", extra_config={"x_bearer_token": "fake-token"})
        with patch.object(crawler, "_fetch_tweet", new_callable=AsyncMock, return_value=payload):
            result = await crawler.crawl("https://x.com/testuser/status/1346889436626259968", config)
        assert result.success is True
        assert "Hello from the API" in result.content
        assert "Test User" in result.content
        assert "@testuser" in result.content
        assert result.metadata.get("tweet_id") == "1346889436626259968"
        assert result.metadata.get("source_type") == "x_tweet"
        await crawler.close()

    async def test_x_crawler_search_mocked(self):
        """search() returns list of CrawlResult from mocked search API."""
        crawler = XCrawler()
        config = CrawlerConfig(
            provider="x",
            extra_config={
                "x_bearer_token": "fake-token",
                "x_scraper_config": XScraperConfig(max_search_results=10, search_max_pages=1).model_dump(),
            },
        )
        payload = {
            "data": [
                {
                    "id": "1",
                    "text": "Hello world",
                    "author_id": "u1",
                    "created_at": "2021-01-01T12:00:00.000Z",
                }
            ],
            "includes": {"users": [{"id": "u1", "username": "alice", "name": "Alice"}]},
            "meta": {},
        }
        with patch.object(crawler, "_search_tweets", new_callable=AsyncMock, return_value=payload):
            results = await crawler.search("hello", config=config)
        assert len(results) == 1
        assert results[0].success is True
        assert "Hello world" in results[0].content
        await crawler.close()

    async def test_x_crawler_scrape_user_mocked(self):
        """scrape_user() returns XUserScrapeResult with user, followers, following."""
        crawler = XCrawler()
        config = CrawlerConfig(
            provider="x",
            extra_config={
                "x_bearer_token": "fake-token",
                "x_scraper_config": XScraperConfig(
                    max_followers_per_user=5,
                    max_following_per_user=5,
                    include_followers=True,
                    include_following=True,
                    graph_depth=1,
                ).model_dump(),
            },
        )
        user = {"id": "u1", "username": "bob", "name": "Bob", "description": "Dev"}
        followers_payload = {"data": [{"id": "f1", "username": "f1", "name": "F1"}], "meta": {}}
        following_payload = {"data": [{"id": "g1", "username": "g1", "name": "G1"}], "meta": {}}
        with patch.object(crawler, "_user_by_username", new_callable=AsyncMock, return_value=user), \
             patch.object(crawler, "_user_followers", new_callable=AsyncMock, return_value=followers_payload), \
             patch.object(crawler, "_user_following", new_callable=AsyncMock, return_value=following_payload):
            result = await crawler.scrape_user("bob", config=config)
        assert isinstance(result, XUserScrapeResult)
        assert result.user["username"] == "bob"
        assert len(result.followers) == 1
        assert result.followers[0]["username"] == "f1"
        assert len(result.following) == 1
        assert result.following[0]["username"] == "g1"
        crawl_results = result.to_crawl_results()
        assert len(crawl_results) >= 1
        assert "Bob" in crawl_results[0].content
        assert crawl_results[0].metadata.get("source_type") == "x_user"
        assert crawl_results[0].metadata.get("x_user_id") == "u1"
        await crawler.close()

    def test_get_x_scraper_config_default(self):
        """_get_x_scraper_config returns default when extra_config has no x_scraper_config."""
        config = CrawlerConfig(provider="x", extra_config={})
        x_config = _get_x_scraper_config(config)
        assert isinstance(x_config, XScraperConfig)
        assert x_config.max_search_results == 100
        assert x_config.graph_depth == 1


class TestCrawlerConfig:
    """Tests for CrawlerConfig."""
    
    def test_crawler_config_defaults(self):
        """Test CrawlerConfig default values."""
        config = CrawlerConfig()
        
        assert config.provider == "crawl4ai"
        assert config.max_depth == 1
        assert config.max_pages == 10
        assert config.wait_for_js is True
        assert config.extract_markdown is True
        assert config.respect_robots_txt is True
    
    def test_crawler_config_custom(self):
        """Test CrawlerConfig custom values."""
        config = CrawlerConfig(
            provider="http",
            max_depth=3,
            max_pages=50,
            wait_for_js=False,
            follow_patterns=[r"example\.com"],
            exclude_patterns=[r"example\.com/admin"],
        )
        
        assert config.provider == "http"
        assert config.max_depth == 3
        assert config.max_pages == 50
        assert config.wait_for_js is False
        assert len(config.follow_patterns) == 1
        assert len(config.exclude_patterns) == 1


class TestCrawlExtractionConfig:
    """Tests for CrawlExtractionConfig."""
    
    def test_extraction_config_defaults(self):
        """Test CrawlExtractionConfig default values."""
        config = CrawlExtractionConfig()
        
        assert config.enabled is True
        assert config.semantic_guide is None
        assert config.extraction_query is None
        assert config.output_schema is None
        assert config.auto_improve is False
        assert config.temperature == 0.0
    
    def test_extraction_config_custom(self):
        """Test CrawlExtractionConfig custom values."""
        config = CrawlExtractionConfig(
            enabled=True,
            semantic_guide="Extract product data",
            auto_improve=True,
            max_improve_iterations=5,
            temperature=0.5,
        )
        
        assert config.enabled is True
        assert config.semantic_guide == "Extract product data"
        assert config.auto_improve is True
        assert config.max_improve_iterations == 5
        assert config.temperature == 0.5


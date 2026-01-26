"""Tests for web crawler components."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from spiderweb.crawlers.base import CrawlResult
from spiderweb.crawlers.http import HttpCrawler
from spiderweb.crawlers.extraction import CrawlExtractor
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
    
    @pytest.mark.asyncio
    async def test_http_crawler_init(self):
        """Test HttpCrawler initialization."""
        crawler = HttpCrawler()
        assert crawler._session is None
        await crawler.close()
    
    @pytest.mark.asyncio
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
    
    @pytest.mark.asyncio
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
    
    @pytest.mark.asyncio
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
    
    @pytest.mark.asyncio
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
    
    @pytest.mark.asyncio
    async def test_web_loader_init(self):
        """Test WebLoader initialization."""
        loader = WebLoader()
        
        assert loader.crawler is not None
        assert loader.crawler_config is not None
        assert loader.extraction_config is not None
    
    @pytest.mark.asyncio
    async def test_web_loader_with_http_provider(self):
        """Test WebLoader with HTTP crawler."""
        config = CrawlerConfig(provider="http")
        loader = WebLoader(crawler_config=config)
        
        assert isinstance(loader.crawler, HttpCrawler)
    
    @pytest.mark.asyncio
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


# Web Crawling Feature

The Spiderweb web crawling feature provides a powerful, extensible system for fetching and processing web content with intelligent LLM-powered extraction capabilities.

## Overview

The crawl feature combines:
- **Pluggable crawlers** - Multiple backend options (crawl4ai, simple HTTP)
- **Intelligent extraction** - LLM-powered structured data extraction with Pydantic schemas
- **Semantic guidance** - Natural language instructions for what to extract
- **Auto-improvement** - Iterative refinement of extraction results
- **RAG integration** - Seamless integration with existing document pipeline

## Quick Start

### Basic Crawl

```python
from spiderweb import Spiderweb
from gluellm import GlueLLM

async with Spiderweb(llm_client=GlueLLM()) as web:
    result = await web.crawl("https://example.com")
    print(result.markdown)
```

### Structured Extraction with Pydantic

```python
from pydantic import BaseModel

class Product(BaseModel):
    name: str
    price: float
    description: str

result = await web.crawl(
    "https://store.com/product",
    output_schema=Product,
    extraction_config=CrawlExtractionConfig(
        semantic_guide="Extract product details",
        auto_improve=True,
    ),
)

print(result.extracted_data.name)
print(result.extracted_data.price)
```

### Crawl and Query (RAG with Fresh Data)

```python
results = await web.crawl_and_query(
    query="What are the latest pricing changes?",
    urls=["https://company.com/pricing"],
    extraction_config=CrawlExtractionConfig(
        semantic_guide="Focus on pricing tiers and recent updates",
    ),
)
```

## Architecture

### Components

1. **Crawler Protocol** (`spiderweb.crawlers.base`)
   - `Crawler` - Protocol defining crawler interface
   - `CrawlResult` - Data structure for crawl results

2. **Crawler Implementations**
   - `HttpCrawler` - Simple HTTP GET requests with aiohttp
   - `Crawl4AICrawler` - Advanced crawler with JavaScript rendering
   - Pluggable design for future crawlers

3. **Extraction Layer** (`spiderweb.crawlers.extraction`)
   - `CrawlExtractor` - LLM-powered structured extraction
   - Schema validation with Pydantic
   - Auto-improvement loop for better accuracy

4. **Web Loader** (`spiderweb.loaders.web_loader`)
   - `WebLoader` - Converts crawled content to Documents
   - Mirrors FileLoader pattern for consistency

5. **Configuration**
   - `CrawlerConfig` - Crawler behavior (depth, rate limiting, etc.)
   - `CrawlExtractionConfig` - Extraction settings (schema, guidance, etc.)

### Data Flow

```
URL → Crawler → CrawlResult → [Optional: Extractor] → Document → Pipeline
```

## Configuration

### CrawlerConfig

```python
from spiderweb.models.config import CrawlerConfig

config = CrawlerConfig(
    provider="crawl4ai",           # "crawl4ai" or "http"
    max_depth=2,                   # Follow links N levels deep
    max_pages=20,                  # Limit total pages crawled
    follow_patterns=[r"docs/"],    # Regex patterns to follow
    exclude_patterns=[r"archive/"], # Patterns to exclude
    wait_for_js=True,              # Wait for JavaScript rendering
    delay_between_requests=1.0,    # Rate limiting
    timeout_seconds=30,            # Request timeout
)
```

### CrawlExtractionConfig

```python
from spiderweb.models.config import CrawlExtractionConfig

config = CrawlExtractionConfig(
    enabled=True,
    semantic_guide="Extract article metadata and key points",
    extraction_query="Focus on author, date, and main arguments",
    output_schema=Article,         # Pydantic model
    auto_improve=True,             # Iterative improvement
    max_improve_iterations=3,
    temperature=0.0,               # LLM temperature
)
```

## API Reference

### Spiderweb.crawl()

```python
async def crawl(
    self,
    url: str | list[str],
    crawler_config: CrawlerConfig | None = None,
    extraction_config: CrawlExtractionConfig | None = None,
    output_schema: type | None = None,
    ingest: bool = False,
) -> CrawlResult | list[CrawlResult] | IngestionResult | BatchIngestionResult
```

**Parameters:**
- `url` - Single URL or list of URLs to crawl
- `crawler_config` - Optional crawler configuration
- `extraction_config` - Optional extraction configuration
- `output_schema` - Optional Pydantic schema for structured extraction
- `ingest` - If True, ingest crawled content into vector store

**Returns:**
- `CrawlResult(s)` if `ingest=False`
- `IngestionResult(s)` if `ingest=True`

### Spiderweb.crawl_and_query()

```python
async def crawl_and_query(
    self,
    query: str,
    urls: list[str],
    crawler_config: CrawlerConfig | None = None,
    extraction_config: CrawlExtractionConfig | None = None,
    top_k: int = 10,
    **query_kwargs,
) -> QueryResult | QueryResultWithContext
```

**Parameters:**
- `query` - Query string
- `urls` - List of URLs to crawl for fresh content
- `crawler_config` - Optional crawler configuration
- `extraction_config` - Optional extraction configuration
- `top_k` - Number of results to return
- `**query_kwargs` - Additional query parameters (filter_dict, context_window, etc.)

**Returns:**
- `QueryResult` with matched chunks from both fresh and existing content

## CLI Usage

```bash
# Basic crawl
spiderweb crawl https://example.com

# Crawl with link following
spiderweb crawl https://docs.example.com --depth 2 --max-pages 20

# Crawl and ingest
spiderweb crawl https://example.com --ingest --store qdrant://localhost:6333/docs

# Crawl with structured extraction
spiderweb crawl https://store.com/product --extract --semantic-guide "Extract product info"

# Crawl without JavaScript rendering (faster)
spiderweb crawl https://example.com --no-js --provider http

# Crawl with patterns
spiderweb crawl https://example.com \
  --depth 3 \
  --follow-pattern "docs/" \
  --exclude-pattern "archive/" \
  --delay 2.0
```

## Use Cases

### 1. Documentation Ingestion

```python
# Crawl documentation and ingest into RAG system
await web.crawl(
    "https://docs.example.com",
    crawler_config=CrawlerConfig(
        max_depth=3,
        follow_patterns=[r"docs\.example\.com"],
    ),
    ingest=True,
)
```

### 2. Product Monitoring

```python
class ProductInfo(BaseModel):
    name: str
    price: float
    availability: str

# Extract structured product data
product = await web.crawl(
    "https://store.com/product/123",
    output_schema=ProductInfo,
    extraction_config=CrawlExtractionConfig(
        semantic_guide="Extract current product information",
        auto_improve=True,
    ),
)
```

### 3. Research with Fresh Data

```python
# Query with fresh web content
results = await web.crawl_and_query(
    query="What are the latest developments in AI safety?",
    urls=[
        "https://openai.com/blog",
        "https://anthropic.com/news",
    ],
    extraction_config=CrawlExtractionConfig(
        semantic_guide="Focus on recent AI safety research and announcements",
    ),
)
```

### 4. Competitive Intelligence

```python
class CompetitorAnalysis(BaseModel):
    pricing_tiers: list[dict]
    new_features: list[str]
    market_positioning: str

analysis = await web.crawl(
    ["https://competitor1.com", "https://competitor2.com"],
    output_schema=CompetitorAnalysis,
    extraction_config=CrawlExtractionConfig(
        semantic_guide="Extract pricing, features, and positioning",
    ),
)
```

## Advanced Features

### Link Following with Patterns

```python
config = CrawlerConfig(
    max_depth=3,
    max_pages=100,
    follow_patterns=[
        r"example\.com/docs",
        r"example\.com/api",
    ],
    exclude_patterns=[
        r"/archive/",
        r"/old-",
        r"\?page=",  # Exclude pagination
    ],
)
```

### Auto-Improvement Loop

The auto-improvement feature iteratively refines extraction results:

```python
config = CrawlExtractionConfig(
    auto_improve=True,
    max_improve_iterations=3,
    improvement_prompt="Ensure all required fields are populated with accurate data",
)

# Extractor will:
# 1. Attempt extraction
# 2. Validate against schema
# 3. If validation fails, refine prompt with error feedback
# 4. Retry up to max_improve_iterations times
```

### Custom Extraction Queries

```python
config = CrawlExtractionConfig(
    semantic_guide="Extract financial data",
    extraction_query="""
    Look for:
    - Revenue figures (in millions)
    - Year-over-year growth percentages
    - Profit margins
    - Geographic breakdown
    Ignore historical data older than 2 years.
    """,
)
```

### Rate Limiting and Politeness

```python
config = CrawlerConfig(
    delay_between_requests=2.0,     # 2 second delay between requests
    max_concurrent=3,               # Max 3 concurrent requests
    respect_robots_txt=True,        # Respect robots.txt
    timeout_seconds=30,
)
```

## Testing

Run the crawler tests:

```bash
pytest tests/test_crawlers.py -v
```

## Performance Considerations

1. **Crawler Choice**
   - Use `HttpCrawler` for static content (faster)
   - Use `Crawl4AICrawler` for JavaScript-heavy sites (more accurate)

2. **Extraction Performance**
   - Schema-guided extraction is slower but more accurate
   - Auto-improvement adds 2-3x latency but improves quality
   - Consider batch processing for many URLs

3. **Rate Limiting**
   - Always set `delay_between_requests` for public sites
   - Use `max_concurrent` to control load
   - Respect `robots.txt` and site ToS

## Troubleshooting

### Extraction Fails

```python
# Try with auto-improvement
config = CrawlExtractionConfig(
    auto_improve=True,
    max_improve_iterations=5,  # More iterations
    temperature=0.0,            # More deterministic
)
```

### JavaScript Content Not Loading

```python
# Use Crawl4AI with JavaScript rendering
config = CrawlerConfig(
    provider="crawl4ai",
    wait_for_js=True,
    timeout_seconds=60,  # Longer timeout for slow sites
)
```

### Too Many Pages Crawled

```python
# Tighter controls
config = CrawlerConfig(
    max_depth=1,        # Don't follow links
    max_pages=1,        # Just the starting URL
    follow_patterns=[],  # No pattern matching
)
```

## Future Enhancements

- [ ] Sitemap parsing for smarter crawling
- [ ] Browser automation backend (Playwright/Selenium)
- [ ] JavaScript execution and interaction
- [ ] Form submission support
- [ ] Authentication handling
- [ ] Distributed crawling
- [ ] Incremental crawling (only fetch changed content)

## Related Documentation

- [Main README](../README.md)
- [Usage Examples](../examples/crawl_usage.py)
- [API Documentation](./API.md)


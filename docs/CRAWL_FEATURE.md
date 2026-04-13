# Web Crawling Feature

> Fetch websites, extract the good stuff, skip the cruft.

The web crawling feature provides a powerful system for fetching and processing web content. Use simple HTTP for static sites, or Playwright-backed rendering for the JavaScript-heavy ones. Add LLM-powered extraction to pull structured data with Pydantic schemas.

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
from superglue import GlueLLM

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
   - `XCrawler` - X (Twitter) API v2: status URLs, search (keywords/hashtags), user followers/following with depth control
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
    provider="crawl4ai",           # "crawl4ai", "http", or "x"
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

### X (Twitter) scraper — search, hashtags, user graph

The `x` crawler uses the X API v2 (pay-per-request). Set `SPIDERWEB_X_BEARER_TOKEN` or pass `extra_config["x_bearer_token"]`.

**1. Search by keywords or hashtags**

```python
from spiderweb.crawlers import XCrawler
from spiderweb.models.config import CrawlerConfig, XScraperConfig

crawler = XCrawler()
config = CrawlerConfig(
    provider="x",
    extra_config={
        "x_bearer_token": "your-token",
        "x_scraper_config": XScraperConfig(
            max_search_results=50,
            search_max_pages=2,
            delay_between_requests=0.5,
        ).model_dump(),
    },
)
results = await crawler.search("python OR #python", config=config)
# results is list[CrawlResult] — one per tweet; ingest as usual
```

**2. Scrape a user: profile, followers, following (with depth control)**

```python
x_config = XScraperConfig(
    max_followers_per_user=100,
    max_following_per_user=100,
    include_followers=True,
    include_following=True,
    graph_depth=2,              # 1 = user + their list; 2+ = recurse into those users
    max_users_per_level=50,     # cap per level to avoid explosion
    delay_between_requests=0.5,
)
config = CrawlerConfig(
    provider="x",
    extra_config={
        "x_bearer_token": "your-token",
        "x_scraper_config": x_config.model_dump(),
    },
)
scrape_result = await crawler.scrape_user("username", config=config)
# scrape_result is XUserScrapeResult: .user, .followers, .following, .levels
# Convert to CrawlResults for ingestion:
for crawl_result in scrape_result.to_crawl_results():
    # feed into your pipeline
    ...
```

**XScraperConfig controls**

| Field | Default | Description |
|-------|---------|-------------|
| `max_search_results` | 100 | Max tweets per search request (10–100) |
| `search_max_pages` | 1 | Max pagination pages per search |
| `max_followers_per_user` | 100 | Max followers to fetch per user (1–1000) |
| `max_following_per_user` | 100 | Max following to fetch per user (1–1000) |
| `include_followers` | True | Include followers when scraping a user |
| `include_following` | True | Include following when scraping a user |
| `graph_depth` | 1 | Levels deep (1 = user + list; 2+ = recurse) |
| `max_users_per_level` | 50 | When depth > 1, max users to expand per level |
| `delay_between_requests` | 0.5 | Seconds between API requests (rate limits) |

### Parsing entities and topics from X (and scoping queries)

X crawl results (tweets and user profiles) are tagged with `source_type` and IDs in metadata. When you ingest them with the **same pipeline as chunks** (chunking + LangExtract + entity-relations + graph store), entities and topics are extracted and stored. You can then **query and scope** by source.

1. **Ingest X content with entity/topic extraction**

   Use `ingest_x_crawl_results()` so each CrawlResult becomes a Document and runs through the full pipeline (including LangExtract and entity-relations add-ons). Enable add-ons when constructing Spiderweb (e.g. `chunk_add_ons=["langextract", "entity_entity_relations"]`) and pass a graph store so entities and relationships are written.

   ```python
   from spiderweb import Spiderweb
   from spiderweb.crawlers import XCrawler
   from spiderweb.models.config import CrawlerConfig, XScraperConfig

   async with Spiderweb(
       llm_client=llm,
       chunk_add_ons=["langextract", "entity_entity_relations"],
       graph_store=neo4j_store,
   ) as web:
       crawler = XCrawler()
       config = CrawlerConfig(provider="x", extra_config={"x_bearer_token": "..."})
       results = await crawler.search("#python", config=config)
       batch = await web.ingest_x_crawl_results(results, crawler_name="XCrawler")
   ```

2. **Scope vector queries**

   Chunks from X documents get `source_type`, `tweet_id`, `x_user_id`, etc. in `chunk.metadata.extra`. Use `filter_dict` when querying the vector store to scope to X only (e.g. `filter_dict={"source_type": "x_tweet"}` or `{"source_type": "x_user"}`), if your store supports it.

3. **Scope graph queries**

   Entities from X documents have the same attributes on the graph node. Use `list_entities(attributes_filter={"source_type": "x_tweet"})` or the CLI:

   ```bash
   spiderweb graph-query entities --graph-store neo4j://... --source-type x_tweet --limit 100
   spiderweb graph-query entities --graph-store neo4j://... --attr source_type=x_user
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

## Related Documentation

- [Main README](../README.md)
- [Usage Examples](../examples/crawl_usage.py)
- [Local Storage](./LOCAL_STORAGE.md)
- [CLI Reference](./CLI.md)
- [Configuration Reference](./CONFIGURATION.md)


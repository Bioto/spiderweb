# Spiderweb

> **TL;DR:** Crawl websites. Extract documents. Chunk intelligently. Store in vectors. Query with superpowers. All without writing the same boilerplate for the hundredth time.

Spiderweb is a production-minded document processing + RAG ingestion pipeline. Feed it PDFs, Office docs, markdown, web pages, or entire websites—it turns that chaos into clean, searchable chunks you can actually use. Think of it as the "I don't want to think about ingestion anymore" library.

## Powered by GlueLLM

Spiderweb uses **[GlueLLM](https://github.com/Bioto/glue-llm)** for embeddings and LLM-powered features (query expansion, extraction, validation). That means you get provider-agnostic models, automatic retries, and sane defaults without wiring up every client yourself.

If you like Spiderweb, you'll probably like GlueLLM too. They're built to work together.

---

## What is this?

Spiderweb takes raw documents and transforms them into searchable, semantically meaningful chunks stored in a vector database. It's for people building RAG systems who want a pipeline that's:

- **Boringly reliable** — good defaults, clear configuration, predictable behavior
- **Extensible** — pluggable chunkers, validators, crawlers, and stores
- **Practical** — includes a CLI, Docker setup, MCP server, REST API, and real examples

**The vibe:** This is what I call a "vibe-coded" library—built iteratively based on real-world needs rather than some grand architectural vision. That said, it's running in production projects, so it's battle-tested enough to be useful.

---

## Why you might like it

- **Web crawling** — static HTTP or Playwright-backed (via crawl4ai), because some sites are stuck in 2010 and need JavaScript rendering
- **Multi-format extraction** — PDF, DOCX, PPTX, XLSX, HTML, markdown, and 20+ other formats via `markitdown`
- **Smart chunking** — hierarchical, semantic, sentence, or sliding window (because one size never fits all)
- **Quality gates** — deduplication and content quality validation, so you're not storing garbage
- **Query expansion** — multi-query reformulation and HyDE with RRF fusion (for when simple queries aren't cutting it)
- **Structured extraction** — Pydantic schemas for pulling structured data from web pages (because we're civilized)
- **Vector store support** — Qdrant (local/cloud) and in-memory for development
- **MCP + REST API** — expose everything to AI assistants or your own services
- **Progressive RAG** — lazy-loading document processing for massive documents
- **Local storage** — save crawled content as markdown, HTML, or JSON for archiving

## Why you might not

- If you want *maximum* low-level control over every step, you might find it "too helpful"
- If you don't want any LLM dependency at all, this won't be your jam (GlueLLM is the engine)
- It's opinionated about how document processing should work

---

## Installation

```bash
# Using uv (recommended)
uv pip install spiderweb

# Using pip
pip install spiderweb
```

### Optional Extras

Install what you need:

```bash
# PDF extraction
pip install "spiderweb[pdf]"

# Office documents (docx, pptx, xlsx)
pip install "spiderweb[office]"

# OCR for scanned PDFs
pip install "spiderweb[ocr]"

# Grounded entity extraction (LangExtract) - source spans and few-shot extraction
pip install "spiderweb[langextract]"

# MCP server for AI assistants
pip install "spiderweb[mcp]"

# REST API server
pip install "spiderweb[api]"

# Everything
pip install "spiderweb[all]"
```

### Docker (recommended for crawling)

Docker includes Playwright + browsers and a Qdrant instance. No local dependency headaches.

```bash
cp env.example .env
# Edit .env with your OPENAI_API_KEY

docker-compose up -d
docker-compose exec spiderweb spiderweb crawl https://example.com
```

See [docker/README.md](docker/README.md) and [SETUP.md](SETUP.md) for details.

---

## Quick Start

### Ingest a Document

```python
import asyncio
from superglue import GlueLLM
from spiderweb import Spiderweb

async def main():
    async with Spiderweb(llm_client=GlueLLM()) as web:
        result = await web.ingest("document.pdf")
        print(f"Chunks created: {result.chunks_created}")
        print(f"Chunks validated: {result.chunks_validated}")
        print(f"Processing time: {result.processing_time_seconds:.2f}s")

asyncio.run(main())
```

### Crawl a Website

```python
from spiderweb.models.config import CrawlerConfig

async with Spiderweb(llm_client=GlueLLM()) as web:
    # Simple single-page crawl
    result = await web.crawl("https://example.com")
    print(f"Title: {result.title}")
    print(f"Links found: {len(result.links)}")
    
    # Crawl with link following
    results = await web.crawl(
        "https://docs.example.com",
        crawler_config=CrawlerConfig(
            max_depth=2,              # Follow links 2 levels deep
            max_pages=50,             # Cap at 50 pages
            follow_patterns=[r"docs/"],  # Only follow /docs/ URLs
            delay_between_requests=1.0,  # Be polite
        ),
    )
    print(f"Crawled {len(results)} pages")
```

### Crawl and Save Locally

```python
# Save to local filesystem (markdown, HTML, JSON, or all)
result = await web.crawl(
    "https://example.com",
    crawler_config=CrawlerConfig(max_depth=2, max_pages=25),
    save_to="./crawled_data",
    save_format="all",  # or "markdown", "html", "json"
)
```

### Crawl and Ingest to Vector Store

```python
async with Spiderweb(
    llm_client=GlueLLM(),
    vector_store_url="qdrant://localhost:6333/my_docs",
) as web:
    result = await web.crawl(
        "https://docs.example.com",
        crawler_config=CrawlerConfig(max_depth=2),
        ingest=True,  # Automatically chunk, embed, and store
    )
    print(f"Ingested {result.total_chunks} chunks")
```

### Structured Extraction with Pydantic

Pull structured data from web pages. Because copy-pasting from websites is for amateurs.

```python
from pydantic import BaseModel, Field

class Product(BaseModel):
    name: str = Field(description="Product name")
    price: float = Field(description="Price in dollars")
    description: str = Field(description="Product description")
    features: list[str] = Field(default_factory=list)

from spiderweb.models.config import CrawlExtractionConfig

result = await web.crawl(
    "https://store.example.com/product/widget-pro",
    output_schema=Product,
    extraction_config=CrawlExtractionConfig(
        semantic_guide="Extract product information",
        auto_improve=True,  # Iteratively improve extraction
    ),
)

print(f"Product: {result.extracted_data.name}")
print(f"Price: ${result.extracted_data.price:.2f}")
```

### Query Your Documents

```python
# Basic query
results = await web.query("What is the main topic?", top_k=5)
for chunk in results.chunks:
    print(chunk["content"][:200])
```

### Query with Expansion (for better recall)

When simple queries miss relevant content, query expansion generates alternative phrasings and combines results.

```python
from spiderweb.models.config import QueryExpansionConfig

results = await web.query(
    "How do I deploy to production?",
    query_expansion=QueryExpansionConfig(
        enabled=True,
        strategy="multi_query",  # or "hyde" for hypothetical answers
        num_expansions=3,
    ),
)

# See what queries were generated
print("Expanded queries:", results.expanded_queries)
# ['How do I deploy to production?',
#  'What is the production deployment process?',
#  'Steps for releasing to live environment']
```

### Query with Context Windows

Retrieve surrounding chunks for better understanding.

```python
from spiderweb.models.config import ContextWindowConfig

results = await web.query(
    "Q4 revenue",
    context_window=ContextWindowConfig(
        enabled=True,
        chunks_before=2,
        chunks_after=2,
        context_mode="page",  # or "chunk"
        semantic_guide="Focus on financial metrics",
    ),
)

# Access context for each match
for idx, match_context in results.context_by_match.items():
    print(f"Match {idx}: {len(match_context.chunks)} context chunks")
```

### Search, Crawl, Extract Pipeline

The power move: search the web, crawl relevant results, and extract what you need.

```python
from spiderweb.models.config import SearchDepthConfig

trace = await web.search_crawl_extract(
    query="python web scraping best practices 2024",
    depth_config=SearchDepthConfig(
        max_search_rounds=2,
        crawl_results_per_round=5,
    ),
    save_to="./research",
)

print(f"Searched {trace.rounds_count} rounds")
print(f"Crawled {len(trace.urls_crawled)} URLs")
for summary in trace.summaries[:3]:
    print(f"- {summary.url}: {summary.summary[:100]}...")
```

---

## Chunking Strategies

Choose the right strategy for your documents:

| Strategy | Best For | Requires LLM | Speed |
|----------|----------|--------------|-------|
| `hierarchical` (default) | Markdown, documentation with headings | No | Fast |
| `semantic` | Any text, topic-based splitting | Yes | Slow |
| `sentence` | Articles, prose | No | Fast |
| `sliding_window` | Any text, prototyping | No | Fast |

```python
from spiderweb.models.config import ChunkerConfig

# Hierarchical (preserves document structure)
config = ChunkerConfig(strategy="hierarchical", max_chunk_size=2000)

# Semantic (uses embeddings to find topic boundaries)
config = ChunkerConfig(strategy="semantic", semantic_threshold=0.7)

# Sentence (never splits mid-sentence)
config = ChunkerConfig(strategy="sentence", max_chunk_size=1000)

# Sliding window (simple and fast)
config = ChunkerConfig(strategy="sliding_window", chunk_overlap=200)
```

See [docs/CHUNKERS.md](docs/CHUNKERS.md) for the full guide.

---

## LangExtract (Grounded Entity Extraction)

With `pip install spiderweb[langextract]` you can run [LangExtract](https://github.com/google/langextract) as a chunk add-on: extract entities with **precise source spans** (character offsets) and optional few-shot examples. Results are stored in `document.metadata.extra["langextract"]` and, when chunks have `start_char`/`end_char`, overlapping entities are attached to each chunk's `metadata.extra["langextract_entities"]`.

Set `LANGEXTRACT_API_KEY` (e.g. for Gemini) or pass `api_key` in options.

```python
from spiderweb.models.config import ChunkAddOnConfig, LangExtractAddOnOptions

# Enable LangExtract add-on with prompt and few-shot examples
opts = LangExtractAddOnOptions(
    prompt_description="Extract people, places, and dates. Use exact text.",
    examples=[
        {
            "text": "On Jan 1, Alice met Bob in Paris.",
            "extractions": [
                {"extraction_class": "person", "extraction_text": "Alice", "attributes": {}},
                {"extraction_class": "person", "extraction_text": "Bob", "attributes": {}},
                {"extraction_class": "place", "extraction_text": "Paris", "attributes": {}},
                {"extraction_class": "date", "extraction_text": "Jan 1", "attributes": {}},
            ],
        },
    ],
    model_id="gemini-2.5-flash",
    extraction_passes=2,  # for long documents
)
config = ChunkAddOnConfig(
    enabled=["langextract"],
    options={"langextract": opts.model_dump()},
)

async with Spiderweb(llm_client=GlueLLM(), chunk_addon_config=config) as web:
    result = await web.ingest("report.pdf")
    extractions = result.document.metadata.extra.get("langextract", {}).get("extractions", [])
    print(f"Extracted {len(extractions)} entities with source grounding")
```

---

## CLI Reference

Spiderweb includes a CLI for when you just want to get things done.

### Crawl

```bash
# Crawl a single page
spiderweb crawl https://example.com

# Crawl with depth and save locally
spiderweb crawl https://docs.example.com \
  --depth 3 \
  --max-pages 100 \
  --save-to ./crawled \
  --save-format markdown

# Crawl and ingest to Qdrant
spiderweb crawl https://example.com \
  --depth 2 \
  --ingest \
  --store qdrant://localhost:6333/docs

# Crawl with structured extraction
spiderweb crawl https://store.com/product \
  --extract \
  --semantic-guide "Extract product name, price, description"

# Fast static crawl (no JavaScript)
spiderweb crawl https://example.com --provider http --no-js
```

### Ingest

```bash
# Ingest a file
spiderweb ingest document.pdf

# Ingest a directory with semantic chunking
spiderweb ingest /path/to/docs --chunker semantic --recursive

# Ingest with custom chunk size
spiderweb ingest document.pdf --chunk-size 2000 --chunk-overlap 400

# Ingest with OCR (for scanned PDFs)
spiderweb ingest scanned.pdf --use-ocr

# Progressive RAG mode (summaries first, full content on-demand)
spiderweb ingest large_document.pdf --progressive
```

### Query

```bash
# Simple query
spiderweb query "What is machine learning?"

# Query with scores and more results
spiderweb query "authentication flow" --show-scores --top-k 10

# Query with expansion
spiderweb query "deployment process" --expand --verbose

# Query with context window
spiderweb query "Q4 revenue" --context-before 3 --context-after 3

# Full-featured query
spiderweb query "API errors" \
  --store qdrant://localhost:6333/docs \
  --top-k 10 \
  --expand \
  --expand-strategy multi_query \
  --context-before 2 \
  --context-after 2 \
  --show-scores \
  --verbose
```

### Search

```bash
# Search the web and crawl results
spiderweb search "python web scraping" \
  --max-rounds 2 \
  --crawl-per-round 5 \
  --save-to ./research

# Save trace for debugging
spiderweb search "AI safety research" \
  --save-trace ./trace.json \
  --trace-format json
```

See [docs/CLI.md](docs/CLI.md) for the complete reference.

---

## MCP Server

Expose Spiderweb to AI assistants via the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/).

### Running the Server

```bash
# Install MCP support
pip install "spiderweb[mcp]"

# For IDE integration (stdio)
spiderweb mcp

# For testing with MCP Inspector (HTTP)
spiderweb mcp --transport streamable-http --port 8000
```

### Available Tools

| Tool | Description |
|------|-------------|
| `crawl_url` | Crawl a single URL, optionally save/ingest |
| `crawl_urls` | Crawl multiple URLs |
| `search_and_crawl` | Search the web and crawl results |

### Adding to Cursor

In Cursor settings, add an MCP server:

```json
{
  "mcpServers": {
    "spiderweb": {
      "command": "spiderweb",
      "args": ["mcp"],
      "transport": "stdio"
    }
  }
}
```

The tools will then be available to the AI assistant.

---

## REST API

Expose crawling and search via FastAPI.

### Running the Server

```bash
# Install API support
pip install "spiderweb[api]"

# Start on default port (8000)
spiderweb api

# Custom host/port
spiderweb api --host 0.0.0.0 --port 8080

# With auto-reload (development)
spiderweb api --reload
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/crawl` | Crawl a single URL |
| POST | `/crawl/batch` | Crawl multiple URLs |
| POST | `/search` | Search the web and crawl results |
| GET | `/health` | Health check |
| GET | `/docs` | Interactive API docs (Swagger UI) |

### Example Usage

```bash
# Crawl a URL
curl -X POST "http://localhost:8000/crawl" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com", "save_format": "markdown"}'

# Search and crawl
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "python tutorials", "max_rounds": 2, "crawl_per_round": 5}'
```

Visit `http://localhost:8000/docs` for interactive API documentation.

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `SPIDERWEB_EMBEDDING_MODEL` | Default embedding model | `text-embedding-3-small` |
| `SPIDERWEB_STORE_URL` | Default vector store URL | `memory://` |
| `SPIDERWEB_LOG_LEVEL` | Logging level | `INFO` |
| `SPIDERWEB_CHUNK_SIZE` | Default chunk size | `1000` |
| `SPIDERWEB_CHUNK_OVERLAP` | Default chunk overlap | `200` |

### Example .env

```bash
OPENAI_API_KEY=sk-...
SPIDERWEB_EMBEDDING_MODEL=text-embedding-3-small
SPIDERWEB_STORE_URL=qdrant://localhost:6333/default
SPIDERWEB_LOG_LEVEL=INFO
```

See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for all configuration options.

---

## Documentation

For the deep dives:

- **[Crawling](docs/CRAWL_FEATURE.md)** — web crawling, extraction, link following
- **[Local Storage](docs/LOCAL_STORAGE.md)** — saving crawled content to disk
- **[Query Features](docs/QUERY.md)** — query expansion, context windows
- **[Chunking Strategies](docs/CHUNKERS.md)** — hierarchical, semantic, sentence, sliding window
- **[Progressive RAG](docs/PROGRESSIVE_RAG.md)** — lazy-loading for massive documents
- **[Extensibility](docs/EXTENSIBILITY.md)** — hooks and custom components
- **[Configuration](docs/CONFIGURATION.md)** — all the knobs and dials
- **[CLI Reference](docs/CLI.md)** — complete command-line reference
- **[Examples](examples/)** — working code you can copy

---

## Contributing

PRs welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).

---

*"I've been writing document processing pipelines for longer than I'd like to admit. Spiderweb is what I wish I'd had from the start."*

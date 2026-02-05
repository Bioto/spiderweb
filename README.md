# Spiderweb

> **TL;DR:** Crawl, extract, chunk, validate, and store documents for RAG — without turning your codebase into a pile of bespoke ingestion scripts.

Spiderweb is a production-minded document processing + RAG ingestion pipeline. Feed it PDFs, Office docs, markdown, web pages, or whole sites — it turns that chaos into clean chunks you can search and retrieve.

## Powered by GlueLLM (yes, on purpose)

Spiderweb uses **[GlueLLM](https://github.com/Bioto/glue-llm)** for embeddings and LLM-powered features (query expansion, extraction, optional validation). That means you get provider-agnostic models, retries, and sane defaults without wiring up every client yourself.

If you like Spiderweb, you’ll probably like GlueLLM too. If you like GlueLLM, Spiderweb is the “put it to work” companion.

## What is this?

Spiderweb takes raw documents and transforms them into searchable, semantically meaningful chunks stored in a vector database (Qdrant by default, in-memory for dev). It’s meant for people building RAG systems who want a pipeline that’s:

- **Boringly reliable**: good defaults, clear configuration, predictable behavior
- **Extensible**: pluggable chunkers, validators, crawlers, and stores
- **Practical**: includes a CLI, Docker setup, and real examples

## Why you might like it

- **Web crawling**: simple HTTP or Playwright-backed crawling (via crawl4ai), plus local file storage
- **Multi-format extraction**: PDF/Office/etc via `markitdown`
- **Smart chunking**: hierarchical, semantic, sliding window, sentence-based
- **Quality gates**: deduplication and content quality validation
- **Vector store support**: Qdrant (local/cloud) and in-memory
- **Query expansion**: multi-query and HyDE (with RRF fusion)

## Why you might not

- If you want *maximum* low-level control over every step, you might find it “too helpful”
- If you don’t want any LLM dependency at all, Spiderweb won’t be your favorite (GlueLLM is the engine)

## Installation

```bash
# Using uv (recommended)
uv pip install spiderweb

# Using pip
pip install spiderweb

# With optional extras
pip install "spiderweb[pdf,office,ocr]"
```

## Quick start

### Option A: Docker (recommended for crawling)

Docker includes Playwright + browsers, and a Qdrant instance for storage:

```bash
cp env.example .env
# Edit .env with OPENAI_API_KEY (and others if you want)

docker-compose up -d

# Crawl
docker-compose exec spiderweb spiderweb crawl https://example.com
```

For the real “Docker quickstart”, see:
- **[docker/README.md](docker/README.md)** (Playwright, Qdrant, CLI examples)
- **[SETUP.md](SETUP.md)** (Docker vs local install guidance)

### Option B: Local

```bash
uv pip install -e ".[dev]"
playwright install chromium
```

## The 60-second API tour

### Ingest a file

```python
import asyncio
from gluellm import GlueLLM
from spiderweb import Spiderweb

async def main():
    async with Spiderweb(llm_client=GlueLLM()) as web:
        result = await web.ingest("document.pdf")
        print(f"Chunks created: {result.chunks_created}")

asyncio.run(main())
```

### Crawl a site and save locally (and optionally ingest)

```python
import asyncio
from gluellm import GlueLLM
from spiderweb import Spiderweb
from spiderweb.models.config import CrawlerConfig

async def main():
    async with Spiderweb(llm_client=GlueLLM()) as web:
        result = await web.crawl(
            url="https://example.com",
            crawler_config=CrawlerConfig(provider="crawl4ai", max_depth=2, max_pages=25),
            save_to="./crawled_data",
            save_format="all",
            ingest=True,
        )
        print(result)

asyncio.run(main())
```

### Query your stored chunks

```python
# Inside the same `async with Spiderweb(...) as web:` block:
results = await web.query("What is the main topic?", top_k=5)
for chunk in results.chunks:
    print(chunk["content"][:120])
```

## CLI (because sometimes you just want it done)

```bash
# Crawl and save locally
spiderweb crawl https://example.com --save-to ./crawled

# Crawl and ingest to Qdrant
spiderweb crawl https://example.com --ingest --store qdrant://localhost:6333/docs

# Query
spiderweb query "What is X?" --store qdrant://localhost:6333/docs --top-k 5
```

## Optional: MCP Server

Spiderweb can expose its crawling and search tools via the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/), allowing AI clients like Cursor, Claude Desktop, or the MCP Inspector to invoke them.

### Installation

```bash
pip install "spiderweb[mcp]"
```

### Running the Server

**For IDE integration (stdio):**
```bash
spiderweb mcp
```

**For testing with MCP Inspector (HTTP):**
```bash
spiderweb mcp --transport streamable-http --port 8000
```

Then connect with: `npx -y @modelcontextprotocol/inspector` → `http://localhost:8000/mcp`

### Available Tools

- **crawl_url**: Crawl a single URL
  - Args: `url`, optional `save_to`, `save_format`, `vector_store_url`
  - Returns: `{url, success, title, markdown_preview, links_count, error}`

- **crawl_urls**: Crawl multiple URLs
  - Args: `urls` (list), optional `save_to`, `save_format`, `vector_store_url`
  - Returns: `{results: [...]}` (list of crawl_url-style results)

- **search_and_crawl**: Search the web and crawl results
  - Args: `query`, optional `max_rounds`, `crawl_per_round`, `save_to`, `save_trace_to`, `vector_store_url`
  - Returns: `{query, rounds_count, urls_crawled, urls_filtered, summaries}`

### Adding to Cursor

In Cursor settings, add an MCP server:
- **Command**: `spiderweb mcp`
- **Transport**: `stdio`

The tools will then be available to the AI assistant.

## Optional: REST API Server

Spiderweb can expose its crawling and search capabilities via a simple REST API using FastAPI.

### Installation

```bash
pip install "spiderweb[api]"
```

### Running the Server

```bash
# Start on default port (8000)
spiderweb api

# Custom host/port
spiderweb api --host 0.0.0.0 --port 8080

# With auto-reload (development)
spiderweb api --reload
```

### API Endpoints

- **POST /crawl** - Crawl a single URL
- **POST /crawl/batch** - Crawl multiple URLs
- **POST /search** - Search the web and crawl results
- **GET /health** - Health check
- **GET /docs** - Interactive API documentation (Swagger UI)

### Example Usage

```bash
# Crawl a URL
curl -X POST "http://localhost:8000/crawl" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com", "save_format": "markdown"}'

# Search and crawl
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "python web scraping", "max_rounds": 2, "crawl_per_round": 5}'
```

Visit `http://localhost:8000/docs` for interactive API documentation.

## Docs (short, useful, and not trying to be a novel)

- **Crawling**: [docs/CRAWL_FEATURE.md](docs/CRAWL_FEATURE.md)
- **Local storage**: [docs/LOCAL_STORAGE.md](docs/LOCAL_STORAGE.md)
- **Query features**: [docs/QUERY.md](docs/QUERY.md) — query expansion, context windows
- **Chunking strategies**: [docs/CHUNKERS.md](docs/CHUNKERS.md)
- **Progressive RAG**: [docs/PROGRESSIVE_RAG.md](docs/PROGRESSIVE_RAG.md) — lazy-loading document processing
- **Extensibility**: [docs/EXTENSIBILITY.md](docs/EXTENSIBILITY.md) — hooks and custom components
- **Configuration**: [docs/CONFIGURATION.md](docs/CONFIGURATION.md)
- **CLI reference**: [docs/CLI.md](docs/CLI.md)
- **Examples**: [examples/](examples/)

## Contributing

PRs welcome. Please read [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

MIT — see [`LICENSE`](LICENSE).



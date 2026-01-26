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

## Docs (short, useful, and not trying to be a novel)

- **Crawling**: [docs/CRAWL_FEATURE.md](docs/CRAWL_FEATURE.md)
- **Local storage**: [docs/LOCAL_STORAGE.md](docs/LOCAL_STORAGE.md)
- **Examples**: [examples/](examples/)

## Contributing

PRs welcome. Please read [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

MIT — see [`LICENSE`](LICENSE).



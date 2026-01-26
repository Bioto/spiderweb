# CLI Reference

Spiderweb provides a command-line interface for common document processing and RAG tasks.

## Installation

The CLI is installed automatically with Spiderweb:

```bash
pip install spiderweb
spiderweb --help
```

---

## Commands Overview

| Command | Description |
|---------|-------------|
| `crawl` | Crawl web pages with optional ingestion |
| `ingest` | Ingest documents into vector store |
| `query` | Query the vector store |
| `progressive-query` | Query using Progressive RAG mode |

---

## spiderweb crawl

Crawl web pages and optionally ingest them into the vector store.

### Basic Usage

```bash
# Crawl a single page
spiderweb crawl https://example.com

# Crawl and follow links
spiderweb crawl https://docs.example.com --depth 2 --max-pages 50
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--depth N` | 1 | Maximum crawl depth (1 = single page) |
| `--max-pages N` | 10 | Maximum pages to crawl |
| `--provider NAME` | crawl4ai | Crawler backend (`crawl4ai` or `http`) |
| `--no-js` | - | Disable JavaScript rendering |
| `--delay N` | 1.0 | Seconds between requests |
| `--follow-pattern REGEX` | - | Only follow URLs matching pattern |
| `--exclude-pattern REGEX` | - | Skip URLs matching pattern |
| `--ingest` | - | Ingest crawled content to vector store |
| `--store URL` | - | Vector store URL |
| `--save-to PATH` | - | Save crawled content locally |
| `--save-format FORMAT` | all | Output format (`markdown`, `html`, `json`, `all`) |
| `--extract` | - | Enable LLM extraction |
| `--semantic-guide TEXT` | - | Guide for extraction |

### Examples

```bash
# Crawl documentation site
spiderweb crawl https://docs.python.org \
  --depth 3 \
  --max-pages 100 \
  --follow-pattern "docs\.python\.org"

# Crawl and save locally
spiderweb crawl https://example.com \
  --save-to ./crawled_data \
  --save-format markdown

# Crawl and ingest to Qdrant
spiderweb crawl https://docs.example.com \
  --depth 2 \
  --ingest \
  --store qdrant://localhost:6333/docs

# Crawl, save locally AND ingest (best of both)
spiderweb crawl https://company.com/docs \
  --save-to ./backup \
  --ingest \
  --store qdrant://localhost:6333/docs

# Crawl with extraction
spiderweb crawl https://store.com/product \
  --extract \
  --semantic-guide "Extract product name, price, and description"

# Fast static crawl (no JavaScript)
spiderweb crawl https://example.com \
  --provider http \
  --no-js
```

---

## spiderweb ingest

Ingest documents from files or directories into the vector store.

### Basic Usage

```bash
# Ingest a single file
spiderweb ingest document.pdf

# Ingest a directory
spiderweb ingest /path/to/docs --recursive
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--chunker NAME` | hierarchical | Chunking strategy (`hierarchical`, `semantic`, `sentence`) |
| `--chunk-size N` | 1000 | Maximum chunk size in characters |
| `--chunk-overlap N` | 200 | Overlap between chunks |
| `--store URL` | - | Vector store URL |
| `--no-validation` | - | Disable chunk validation |
| `--recursive/--no-recursive` | recursive | Process subdirectories |
| `--embedding-model NAME` | - | Override embedding model |
| `--use-ocr` | - | Use OCR for PDF extraction |
| `--progressive` | - | Use Progressive RAG mode |
| `--summary-strategy NAME` | first_n_chars | Summary strategy for progressive mode |

### Examples

```bash
# Ingest with semantic chunking
spiderweb ingest /path/to/docs --chunker semantic

# Ingest to specific Qdrant collection
spiderweb ingest document.pdf --store qdrant://localhost:6333/my_docs

# Ingest with custom chunk size
spiderweb ingest document.pdf --chunk-size 2000 --chunk-overlap 400

# Ingest with OCR (for scanned PDFs)
spiderweb ingest scanned.pdf --use-ocr

# Progressive RAG mode (summaries first)
spiderweb ingest large_document.pdf \
  --progressive \
  --summary-strategy llm_summary \
  --store qdrant://localhost:6333/docs
```

---

## spiderweb query

Query the vector store for relevant content.

### Basic Usage

```bash
# Simple query
spiderweb query "What is machine learning?"

# Query specific collection
spiderweb query "How to deploy?" --store qdrant://localhost:6333/docs
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--store URL` | - | Vector store URL |
| `--top-k N` | 5 | Number of results |
| `--embedding-model NAME` | - | Override embedding model |
| `--show-scores` | - | Display similarity scores |
| `--context-before N` | - | Chunks before each match |
| `--context-after N` | - | Chunks after each match |
| `--context-mode MODE` | page | Context mode (`page` or `chunk`) |
| `--semantic-guide TEXT` | - | Guide for context scoring |
| `--expand` | - | Enable query expansion |
| `--expand-strategy NAME` | multi_query | Expansion strategy (`multi_query` or `hyde`) |
| `--expand-num N` | 3 | Number of query expansions |
| `--expand-prompt TEXT` | - | Custom expansion prompt |
| `--verbose` | - | Show expanded queries and details |

### Examples

```bash
# Query with scores
spiderweb query "authentication flow" --show-scores

# Query with more results
spiderweb query "Python best practices" --top-k 10

# Query with context window
spiderweb query "revenue Q4" \
  --context-before 3 \
  --context-after 3 \
  --context-mode page

# Query with semantic guidance
spiderweb query "company performance" \
  --context-before 2 \
  --context-after 2 \
  --semantic-guide "financial metrics and KPIs"

# Query with expansion
spiderweb query "machine learning" --expand --verbose

# Query with HyDE expansion
spiderweb query "memory leaks" \
  --expand \
  --expand-strategy hyde \
  --expand-num 2

# Full-featured query
spiderweb query "deployment process" \
  --store qdrant://localhost:6333/docs \
  --top-k 10 \
  --expand \
  --context-before 2 \
  --context-after 2 \
  --verbose \
  --show-scores
```

---

## spiderweb progressive-query

Query using Progressive RAG mode, which queries page summaries first and triggers full processing on-demand.

### Basic Usage

```bash
spiderweb progressive-query "What is the revenue?" --source-file doc.pdf
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--store URL` | - | Base vector store URL |
| `--top-k N` | 5 | Number of results |
| `--source-file PATH` | - | Original PDF for on-demand processing |
| `--embedding-model NAME` | - | Override embedding model |
| `--use-ocr` | - | Use OCR for page processing |

### Examples

```bash
# Progressive query with source file
spiderweb progressive-query "Explain the strategy" \
  --store qdrant://localhost:6333/adobe_docs \
  --source-file docs/10K_2024.pdf

# Progressive query with OCR
spiderweb progressive-query "What are the key findings?" \
  --source-file scanned_report.pdf \
  --use-ocr
```

---

## Environment Variables

Configure defaults via environment variables:

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key (required) |
| `SPIDERWEB_EMBEDDING_MODEL` | Default embedding model |
| `SPIDERWEB_STORE_URL` | Default vector store URL |
| `SPIDERWEB_LOG_LEVEL` | Logging level (DEBUG, INFO, etc.) |

### Example .env file

```bash
OPENAI_API_KEY=sk-...
SPIDERWEB_EMBEDDING_MODEL=text-embedding-3-small
SPIDERWEB_STORE_URL=qdrant://localhost:6333/default
SPIDERWEB_LOG_LEVEL=INFO
```

---

## Docker Usage

When using Docker, commands are run via `docker-compose exec`:

```bash
# Crawl
docker-compose exec spiderweb spiderweb crawl https://example.com

# Ingest
docker-compose exec spiderweb spiderweb ingest /app/data/document.pdf

# Query
docker-compose exec spiderweb spiderweb query "What is this about?"

# Save to mounted directory
docker-compose exec spiderweb spiderweb crawl https://example.com \
  --save-to /app/data/crawled
```

---

## Makefile Shortcuts

The project includes a Makefile with common shortcuts:

```bash
# Crawl
make crawl URL=https://example.com

# Ingest
make ingest PATH=./docs

# Query
make query Q="What is machine learning?"
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |

---

## Related Documentation

- [Crawl Feature](./CRAWL_FEATURE.md) - Detailed crawling documentation
- [Local Storage](./LOCAL_STORAGE.md) - Saving crawled content
- [Query Features](./QUERY.md) - Query expansion and context
- [Configuration](./CONFIGURATION.md) - All configuration options

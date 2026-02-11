# Spiderweb Examples

Working code examples demonstrating Spiderweb's capabilities. Each example is self-contained and can be run independently.

## Examples

### `basic_spiderweb_usage.py`

Basic document ingestion and querying workflow.

**What it demonstrates:**
- Single file ingestion
- Directory ingestion
- Querying the vector store
- Query expansion for improved recall

**Run it:**
```bash
python examples/basic_spiderweb_usage.py
```

**Requirements:**
- `OPENAI_API_KEY` environment variable set
- A document file or directory to ingest

---

### `crawl_usage.py`

Web crawling examples with various configurations.

**What it demonstrates:**
- Single-page crawling
- Multi-page crawling with link following
- Crawling with custom patterns
- Crawling and ingesting to vector store

**Run it:**
```bash
python examples/crawl_usage.py
```

**Requirements:**
- `OPENAI_API_KEY` environment variable set
- Internet connection for web crawling

---

### `crawl_with_local_storage.py`

Crawling with local file storage for archiving.

**What it demonstrates:**
- Crawling and saving to local filesystem
- Saving in multiple formats (markdown, HTML, JSON)
- Creating an index of crawled content
- Combining local storage with vector store ingestion

**Run it:**
```bash
python examples/crawl_with_local_storage.py
```

**Requirements:**
- `OPENAI_API_KEY` environment variable set
- Internet connection for web crawling
- Write permissions for saving files

---

### `x_search_and_follow_graph.py`

Search X (Twitter) for a term, take the top Y tweets, then expand to all posting users and their followers/following.

**What it demonstrates:**
- X search by keyword/hashtag (`XCrawler.search`)
- Taking top Y results and extracting posting usernames
- Scraping each user: profile, followers, following (`XCrawler.scrape_user`)
- Optional graph depth (1 = user + lists; 2+ = recurse into those users)
- Building a single list of CrawlResults for ingestion (entities, vector/graph store)

**Run it:**
```bash
export SPIDERWEB_X_BEARER_TOKEN=your_bearer_token
python examples/x_search_and_follow_graph.py
```

**Config (edit the script):** `SEARCH_QUERY`, `TOP_Y_TWEETS`, `MAX_FOLLOWERS_PER_USER`, `MAX_FOLLOWING_PER_USER`, `GRAPH_DEPTH`, `MAX_USERS_PER_LEVEL`.

**Requirements:**
- `SPIDERWEB_X_BEARER_TOKEN` (X API v2 Bearer token, pay-per-request)

---

## Running Examples

All examples use async/await, so they must be run with Python's asyncio support:

```bash
# Direct execution
python examples/basic_spiderweb_usage.py

# Or with uv
uv run python examples/basic_spiderweb_usage.py
```

## Configuration

Most examples use default settings. To customize:

1. Set environment variables (see `env.example`)
2. Modify the example code to pass custom configs
3. Use a `.env` file in the project root

## Common Patterns

### Basic Ingestion
```python
from gluellm import GlueLLM
from spiderweb import Spiderweb

async with Spiderweb(llm_client=GlueLLM()) as web:
    result = await web.ingest("document.pdf")
```

### Querying
```python
results = await web.query("What is this document about?", top_k=5)
```

### Crawling
```python
result = await web.crawl("https://example.com")
```

See the individual example files for more detailed patterns and use cases.

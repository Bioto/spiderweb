# Configuration Reference

Complete reference for all Spiderweb configuration options.

## Overview

Spiderweb uses Pydantic models for configuration. All configs have sensible defaults and can be customized via:
- Python API (passing config objects)
- CLI flags
- Environment variables

---

## ChunkerConfig

Controls how documents are split into chunks.

```python
from spiderweb.models.config import ChunkerConfig

config = ChunkerConfig(
    strategy="hierarchical",
    max_chunk_size=1000,
    chunk_overlap=200,
    min_chunk_size=100,
    respect_boundaries=True,
    semantic_threshold=0.7,
    preserve_structure=True,
    max_hierarchy_depth=5,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `strategy` | str/ChunkType | `hierarchical` | Chunking strategy: `hierarchical`, `semantic`, `sentence`, `sliding_window`, or custom |
| `max_chunk_size` | int | 1000 | Maximum chunk size in characters (100-10000) |
| `chunk_overlap` | int | 200 | Overlap between chunks (0-1000) |
| `min_chunk_size` | int | 100 | Minimum chunk size (10-1000) |
| `respect_boundaries` | bool | True | Split at sentence boundaries |
| `semantic_threshold` | float | 0.7 | Similarity threshold for semantic chunking (0-1) |
| `preserve_structure` | bool | True | Keep section structure in hierarchical mode |
| `max_hierarchy_depth` | int | 5 | Max heading depth for hierarchical mode (1-10) |

---

## ValidatorConfig

Controls chunk quality validation.

```python
from spiderweb.models.config import ValidatorConfig

config = ValidatorConfig(
    enable_validation=True,
    min_quality_score=0.3,
    enable_deduplication=True,
    deduplication_threshold=0.95,
    enable_llm_validation=False,
    llm_validation_sample_rate=0.1,
    check_completeness=True,
    check_information_density=True,
    min_information_density=0.2,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_validation` | bool | True | Enable chunk validation |
| `min_quality_score` | float | 0.3 | Minimum quality score (0-1) |
| `enable_deduplication` | bool | True | Remove duplicate chunks |
| `deduplication_threshold` | float | 0.95 | Similarity threshold for dedup (0-1) |
| `enable_llm_validation` | bool | False | Use LLM for coherence validation |
| `llm_validation_sample_rate` | float | 0.1 | Fraction of chunks to LLM-validate (0-1) |
| `check_completeness` | bool | True | Check for complete sentences |
| `check_information_density` | bool | True | Check meaningful content ratio |
| `min_information_density` | float | 0.2 | Minimum density score (0-1) |

---

## VectorStoreConfig

Controls vector store connection.

```python
from spiderweb.models.config import VectorStoreConfig

config = VectorStoreConfig(
    provider="qdrant",
    host="localhost",
    port=6333,
    collection_name="spiderweb_documents",
    api_key=None,
    use_https=False,
    embedding_dimension=1536,
    distance_metric="cosine",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | str | `memory` | Vector store: `memory` or `qdrant` |
| `host` | str | `localhost` | Vector store host |
| `port` | int | 6333 | Vector store port |
| `collection_name` | str | `spiderweb_documents` | Collection/index name |
| `api_key` | str/None | None | API key for cloud deployments |
| `use_https` | bool | False | Use HTTPS connection |
| `embedding_dimension` | int | 1536 | Embedding vector dimension |
| `distance_metric` | str | `cosine` | Distance metric: `cosine`, `euclidean`, `dot` |
| `extra_config` | dict | {} | Provider-specific settings |

### URL Format

Instead of using `VectorStoreConfig`, you can pass a URL string:

```python
web = Spiderweb(
    llm_client=llm,
    vector_store_url="qdrant://localhost:6333/my_collection",
)
```

---

## CrawlerConfig

Controls web crawling behavior.

```python
from spiderweb.models.config import CrawlerConfig

config = CrawlerConfig(
    provider="crawl4ai",
    max_depth=2,
    max_pages=20,
    follow_patterns=[r"docs/"],
    exclude_patterns=[r"archive/"],
    respect_robots_txt=True,
    wait_for_js=True,
    timeout_seconds=30,
    extract_markdown=True,
    delay_between_requests=1.0,
    max_concurrent=5,
    user_agent=None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | str | `crawl4ai` | Crawler backend: `crawl4ai`, `http`, or custom |
| `max_depth` | int | 1 | Crawl depth (1 = single page, 1-10) |
| `max_pages` | int | 10 | Maximum pages to crawl (1-1000) |
| `follow_patterns` | list[str] | [] | Regex patterns for URLs to follow |
| `exclude_patterns` | list[str] | [] | Regex patterns for URLs to skip |
| `respect_robots_txt` | bool | True | Honor robots.txt |
| `wait_for_js` | bool | True | Wait for JavaScript (crawl4ai only) |
| `timeout_seconds` | int | 30 | Request timeout (1-300) |
| `extract_markdown` | bool | True | Convert HTML to markdown |
| `delay_between_requests` | float | 1.0 | Delay between requests in seconds (0-10) |
| `max_concurrent` | int | 5 | Max concurrent requests (1-50) |
| `user_agent` | str/None | None | Custom user agent |
| `extra_config` | dict | {} | Provider-specific settings |

---

## CrawlExtractionConfig

Controls LLM-powered extraction from crawled content.

```python
from spiderweb.models.config import CrawlExtractionConfig

config = CrawlExtractionConfig(
    enabled=True,
    semantic_guide="Extract product information",
    extraction_query="Focus on price and description",
    output_schema=MyPydanticModel,
    auto_improve=True,
    max_improve_iterations=3,
    improvement_prompt=None,
    include_raw_content=False,
    temperature=0.0,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | True | Enable LLM extraction |
| `semantic_guide` | str/None | None | High-level extraction description |
| `extraction_query` | str/None | None | Specific extraction instruction |
| `output_schema` | type/None | None | Pydantic model for structured output |
| `auto_improve` | bool | False | Iteratively improve extraction |
| `max_improve_iterations` | int | 3 | Max improvement iterations (1-10) |
| `improvement_prompt` | str/None | None | Custom improvement prompt |
| `include_raw_content` | bool | False | Include raw HTML in metadata |
| `temperature` | float | 0.0 | LLM temperature (0-2) |

---

## QueryExpansionConfig

Controls query expansion for improved recall.

```python
from spiderweb.models.config import QueryExpansionConfig

config = QueryExpansionConfig(
    enabled=True,
    strategy="multi_query",
    num_expansions=3,
    include_original=True,
    rrf_k=60,
    custom_prompt=None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | False | Enable query expansion |
| `strategy` | str | `multi_query` | Strategy: `multi_query` or `hyde` |
| `num_expansions` | int | 3 | Number of expansions (1-10) |
| `include_original` | bool | True | Include original query |
| `rrf_k` | int | 60 | RRF constant (higher = less aggressive) |
| `custom_prompt` | str/None | None | Custom expansion prompt |

---

## ContextWindowConfig

Controls context retrieval for query results.

```python
from spiderweb.models.config import ContextWindowConfig

config = ContextWindowConfig(
    enabled=True,
    chunks_before=2,
    chunks_after=2,
    context_mode="page",
    semantic_guide=None,
    semantic_boost_weight=0.3,
    deduplicate=True,
    include_match_in_context=True,
    expand_on_low_score=True,
    semantic_min_score=0.5,
    max_expansion_steps=3,
    expansion_step_size=2,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | True | Enable context retrieval |
| `chunks_before` | int | 2 | Chunks/pages before each match |
| `chunks_after` | int | 2 | Chunks/pages after each match |
| `context_mode` | str | `page` | Mode: `page` or `chunk` |
| `semantic_guide` | str/None | None | Topic for semantic scoring |
| `semantic_boost_weight` | float | 0.3 | Semantic score weight (0-1) |
| `deduplicate` | bool | True | Remove overlapping context |
| `include_match_in_context` | bool | True | Include matched chunk |
| `expand_on_low_score` | bool | True | Expand on low semantic scores |
| `semantic_min_score` | float | 0.5 | Threshold for expansion (0-1) |
| `max_expansion_steps` | int | 3 | Max expansion attempts |
| `expansion_step_size` | int | 2 | Chunks per expansion step |

---

## ExtractorConfig

Controls document extraction.

```python
from spiderweb.models.config import ExtractorConfig

config = ExtractorConfig(
    primary_method="markitdown",
    enable_cross_extraction=False,
    cross_extraction_methods=["markitdown"],
    preserve_formatting=True,
    extract_metadata=True,
    extract_images=False,
    ocr_images=False,
    language=None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `primary_method` | str | `markitdown` | Primary extraction method |
| `enable_cross_extraction` | bool | False | Use multiple extractors |
| `cross_extraction_methods` | list[str] | ["markitdown"] | Methods for cross-extraction |
| `preserve_formatting` | bool | True | Keep document formatting |
| `extract_metadata` | bool | True | Extract document metadata |
| `extract_images` | bool | False | Extract embedded images |
| `ocr_images` | bool | False | OCR images (requires dependencies) |
| `language` | str/None | None | Document language (auto-detect) |

---

## BatchConfig

Controls batch processing.

```python
from spiderweb.models.config import BatchConfig

config = BatchConfig(
    max_concurrent_extractions=5,
    max_concurrent_embeddings=10,
    batch_size=100,
    continue_on_error=True,
    save_checkpoint_interval=100,
    checkpoint_path=None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_concurrent_extractions` | int | 5 | Max concurrent file extractions |
| `max_concurrent_embeddings` | int | 10 | Max concurrent embedding requests |
| `batch_size` | int | 100 | Batch size for vector store ops |
| `continue_on_error` | bool | True | Continue on individual failures |
| `save_checkpoint_interval` | int | 100 | Checkpoint interval (0 = disabled) |
| `checkpoint_path` | str/None | None | Checkpoint file path |

---

## Environment Variables

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

---

## Combining Configurations

```python
from gluellm import GlueLLM
from spiderweb import Spiderweb
from spiderweb.models.config import (
    ChunkerConfig,
    ValidatorConfig,
    VectorStoreConfig,
    QueryExpansionConfig,
    ContextWindowConfig,
)

# Create with multiple configs
web = Spiderweb(
    llm_client=GlueLLM(),
    chunker_config=ChunkerConfig(
        strategy="semantic",
        max_chunk_size=1500,
    ),
    validator_config=ValidatorConfig(
        enable_deduplication=True,
        min_quality_score=0.4,
    ),
    store_config=VectorStoreConfig(
        provider="qdrant",
        host="localhost",
        port=6333,
        collection_name="my_docs",
    ),
)

# Query with expansion and context
result = await web.query(
    "How does authentication work?",
    query_expansion=QueryExpansionConfig(
        enabled=True,
        strategy="multi_query",
    ),
    context_window=ContextWindowConfig(
        enabled=True,
        chunks_before=3,
        chunks_after=3,
    ),
)
```

---

## Related Documentation

- [Chunkers](./CHUNKERS.md) - Detailed chunking strategies
- [Query Features](./QUERY.md) - Query expansion and context
- [CLI Reference](./CLI.md) - CLI options mapping to configs
- [Extensibility](./EXTENSIBILITY.md) - Custom components

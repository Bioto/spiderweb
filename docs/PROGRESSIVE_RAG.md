# Progressive RAG

> Why process 500 pages when you only ever query 20 of them?

Progressive RAG is a lazy-loading strategy that creates lightweight page summaries first and fully processes pages on-demand as they're queried. Perfect for massive documents where you'd rather not wait an hour for initial ingestion.

## Overview

Traditional RAG ingestion processes entire documents upfront:
1. Extract all pages → 2. Chunk everything → 3. Embed all chunks → 4. Store

Progressive RAG defers most work until query time:
1. Extract pages → 2. Create summaries → 3. Embed summaries → 4. Store summaries
5. *Query hits summary* → 6. Full process just that page → 7. Cache for future queries

### Benefits

| Metric | Traditional | Progressive |
|--------|-------------|-------------|
| Initial ingestion | Slow (all pages) | Fast (summaries only) |
| Storage size | Large (all chunks) | Small initially, grows on demand |
| First query | Instant | Slightly slower (triggers processing) |
| Subsequent queries | Instant | Instant (cached) |

### When to Use

- Large documents (100+ pages)
- Documents where only portions are queried
- Time-sensitive initial ingestion
- Limited storage with selective caching

---

## Architecture

Progressive RAG uses two vector stores:

```
┌─────────────────────────────────────────────────────────┐
│                    Document                              │
└─────────────────────────────────────────────────────────┘
                          │
                    Initial Ingest
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│               Summary Store (_summaries)                 │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │
│  │ Page 1  │ │ Page 2  │ │ Page 3  │ │ Page N  │       │
│  │ Summary │ │ Summary │ │ Summary │ │ Summary │       │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘       │
└─────────────────────────────────────────────────────────┘
                          │
                    Query matches
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  Full Store (_full)                      │
│  ┌──────────────────┐                                    │
│  │ Page 2 Chunks    │  ← Processed on-demand            │
│  │ (full content)   │                                    │
│  └──────────────────┘                                    │
└─────────────────────────────────────────────────────────┘
```

---

## Quick Start

### CLI Ingestion

```bash
# Ingest with progressive mode
spiderweb ingest document.pdf \
  --progressive \
  --summary-strategy first_n_chars \
  --store qdrant://localhost:6333/docs
```

### CLI Query

```bash
# Query with on-demand processing
spiderweb progressive-query "What is the revenue?" \
  --store qdrant://localhost:6333/docs \
  --source-file document.pdf
```

### Python API

```python
import asyncio
from gluellm import GlueLLM
from spiderweb import Spiderweb
from spiderweb.models.config import ChunkerConfig
from spiderweb.models.progressive import ProgressiveRAGConfig, SummaryStrategy
from spiderweb.pipeline.progressive import ProgressiveRAGProcessor


async def progressive_rag_example():
    llm = GlueLLM()
    
    # Create separate stores for summaries and full content
    summary_web = Spiderweb(
        llm_client=llm,
        vector_store_url="qdrant://localhost:6333/docs_summaries",
    )
    
    full_web = Spiderweb(
        llm_client=llm,
        vector_store_url="qdrant://localhost:6333/docs_full",
    )
    
    # Create progressive processor
    config = ProgressiveRAGConfig(
        summary_strategy=SummaryStrategy.FIRST_N_CHARS,
        summary_length=500,
    )
    
    processor = ProgressiveRAGProcessor(
        llm_client=llm,
        summary_store=summary_web.document_processor.vector_store,
        full_store=full_web.document_processor.vector_store,
        document_processor=full_web.document_processor,
        config=config,
    )
    
    # Ingest with summaries only (fast!)
    result = await processor.ingest_with_summaries("large_document.pdf")
    print(f"Created {result['pages_summarized']} page summaries")
    
    # Query - triggers full processing for matched pages
    query_result = await processor.query(
        "What are the key findings?",
        top_k=5,
        source_file="large_document.pdf",
    )
    
    print(f"Found {len(query_result.summary_results)} matching pages")
    print(f"Newly processed: {query_result.newly_processed_pages}")


asyncio.run(progressive_rag_example())
```

---

## Summary Strategies

### first_n_chars (Default)

Extracts the first N characters of each page. Fast and simple.

```python
config = ProgressiveRAGConfig(
    summary_strategy=SummaryStrategy.FIRST_N_CHARS,
    summary_length=500,  # First 500 chars
)
```

**Best for:** Documents with informative beginnings (reports, articles)

### llm_summary

Uses an LLM to generate a 2-3 sentence summary. Higher quality but slower.

```python
config = ProgressiveRAGConfig(
    summary_strategy=SummaryStrategy.LLM_SUMMARY,
    llm_summary_model="openai:gpt-5.1",
)
```

**Best for:** Dense technical documents, research papers

### metadata_only

Extracts headings and key terms from each page. Fastest option.

```python
config = ProgressiveRAGConfig(
    summary_strategy=SummaryStrategy.METADATA_ONLY,
)
```

**Best for:** Well-structured documents with clear headings

---

## Processing Triggers

Control when full page processing occurs:

### IMMEDIATE (Default)

Process pages immediately when a query matches their summary.

```python
config = ProgressiveRAGConfig(
    processing_trigger=ProcessingTrigger.IMMEDIATE,
)
```

### MANUAL

Never auto-process; require explicit processing calls.

```python
config = ProgressiveRAGConfig(
    processing_trigger=ProcessingTrigger.MANUAL,
)

# Later, manually process specific pages
await processor._fully_process_page(page_id, document_id, page_number, file_path)
```

### THRESHOLD

Process only when similarity score exceeds a threshold (not yet implemented).

---

## Configuration Reference

```python
class ProgressiveRAGConfig:
    # Summary generation
    summary_strategy: SummaryStrategy = SummaryStrategy.FIRST_N_CHARS
    summary_length: int = 500
    llm_summary_model: str | None = None
    
    # Processing behavior
    processing_trigger: ProcessingTrigger = ProcessingTrigger.IMMEDIATE
    
    # Full processing settings
    chunk_processed_pages: bool = True
    validate_processed_pages: bool = True


class SummaryStrategy(Enum):
    FIRST_N_CHARS = "first_n_chars"
    LLM_SUMMARY = "llm_summary"
    METADATA_ONLY = "metadata_only"


class ProcessingTrigger(Enum):
    IMMEDIATE = "immediate"
    MANUAL = "manual"
    THRESHOLD = "threshold"
```

---

## OCR Support

For scanned PDFs or documents with broken text encoding:

```python
processor = ProgressiveRAGProcessor(
    llm_client=llm,
    summary_store=summary_store,
    full_store=full_store,
    document_processor=doc_processor,
    config=config,
    use_ocr=True,
    ocr_dpi=150,  # Higher = better quality, slower
)
```

CLI usage:
```bash
spiderweb ingest scanned.pdf --progressive --use-ocr
spiderweb progressive-query "findings" --source-file scanned.pdf --use-ocr
```

**Requirements:**
```bash
pip install pytesseract pillow pymupdf
# Also install Tesseract system package
```

---

## Query Results

The `ProgressiveQueryResult` contains:

```python
@dataclass
class ProgressiveQueryResult:
    query: str                          # Original query
    summary_results: list[PageSummary]  # Matched page summaries
    full_results: list[Chunk]           # Full chunks (if processed)
    newly_processed_pages: list[int]    # Pages processed this query
    processing_time_ms: float           # Total query time
    cache_hit: bool                     # True if results from cache
```

### Accessing Results

```python
result = await processor.query("revenue", top_k=5, source_file=path)

# Matched summaries
for ps in result.summary_results:
    print(f"Page {ps.page_number}: {ps.summary_text[:100]}...")
    print(f"  Fully processed: {ps.is_fully_processed}")

# Full chunks (from newly or previously processed pages)
for chunk in result.full_results:
    print(f"Chunk: {chunk.content[:100]}...")

# Processing info
print(f"Cache hit: {result.cache_hit}")
print(f"Newly processed: {result.newly_processed_pages}")
```

---

## Context Windows with Progressive RAG

Retrieve surrounding page context for matches:

```python
from spiderweb.models.config import ContextWindowConfig

result = await processor.query(
    "revenue Q4",
    top_k=5,
    source_file=path,
    context_window=ContextWindowConfig(
        enabled=True,
        chunks_before=1,  # 1 page before
        chunks_after=1,   # 1 page after
        context_mode="page",  # Required for progressive RAG
    ),
)
```

---

## Complete Example

```python
import asyncio
from pathlib import Path
from gluellm import GlueLLM
from spiderweb import Spiderweb
from spiderweb.models.config import ChunkerConfig, ContextWindowConfig
from spiderweb.models.progressive import ProgressiveRAGConfig, SummaryStrategy
from spiderweb.pipeline.progressive import ProgressiveRAGProcessor


async def main():
    pdf_path = Path("annual_report.pdf")
    llm = GlueLLM()
    
    # Setup stores
    summary_web = Spiderweb(
        llm_client=llm,
        vector_store_url="qdrant://localhost:6333/report_summaries",
        chunker_config=ChunkerConfig(strategy="hierarchical"),
    )
    
    full_web = Spiderweb(
        llm_client=llm,
        vector_store_url="qdrant://localhost:6333/report_full",
        chunker_config=ChunkerConfig(strategy="hierarchical"),
    )
    
    # Configure progressive processor
    config = ProgressiveRAGConfig(
        summary_strategy=SummaryStrategy.LLM_SUMMARY,
    )
    
    processor = ProgressiveRAGProcessor(
        llm_client=llm,
        summary_store=summary_web.document_processor.vector_store,
        full_store=full_web.document_processor.vector_store,
        document_processor=full_web.document_processor,
        config=config,
    )
    
    # Step 1: Fast initial ingestion
    print("Creating page summaries...")
    ingest_result = await processor.ingest_with_summaries(pdf_path)
    print(f"  Summarized {ingest_result['pages_summarized']} pages")
    print(f"  Time: {ingest_result['elapsed_time_seconds']:.2f}s")
    
    # Step 2: Query with on-demand processing
    print("\nQuerying...")
    query_result = await processor.query(
        "What were the Q4 financial results?",
        top_k=3,
        source_file=pdf_path,
        context_window=ContextWindowConfig(
            enabled=True,
            chunks_before=1,
            chunks_after=1,
        ),
    )
    
    print(f"  Found {len(query_result.summary_results)} matching pages")
    print(f"  Query time: {query_result.processing_time_ms:.0f}ms")
    
    if query_result.newly_processed_pages:
        print(f"  Newly processed pages: {query_result.newly_processed_pages}")
    
    # Display results
    for ps in query_result.summary_results:
        print(f"\nPage {ps.page_number}:")
        print(f"  {ps.summary_text[:200]}...")


asyncio.run(main())
```

---

## Performance Considerations

1. **Initial Ingestion**
   - `first_n_chars`: ~1-2 seconds per 100 pages
   - `llm_summary`: ~30-60 seconds per 100 pages (LLM calls)
   - `metadata_only`: <1 second per 100 pages

2. **Query Latency**
   - Summary query: ~100-200ms
   - + On-demand processing: ~500-2000ms per page (depends on OCR, chunking)
   - Cached pages: ~100-200ms

3. **Storage**
   - Summary store: ~1-2KB per page
   - Full store: ~5-20KB per page (grows on demand)

---

## Related Documentation

- [Query Features](./QUERY.md) - Context windows and query expansion
- [Chunkers](./CHUNKERS.md) - Chunking strategies for full processing
- [Configuration](./CONFIGURATION.md) - All configuration options
- [CLI Reference](./CLI.md) - Command-line usage

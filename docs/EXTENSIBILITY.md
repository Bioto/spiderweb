# Extensibility Guide

> Need to do something weird? We've got you covered.

Spiderweb is designed to get out of your way when you need to customize things. This guide covers the two main extension mechanisms: **Component Registries** (plug in your own chunkers, crawlers, extractors) and **Hooks** (intercept data at any pipeline stage).

## Overview

Spiderweb uses a plugin-style architecture with:

- **Registries** - Register custom chunkers, crawlers, and extractors by name
- **Hooks** - Add callbacks at key pipeline stages (before/after extract, chunk, crawl)

All defaults continue to work out of the box. Extensions are opt-in.

## Component Registries

### Available Registries

| Registry | Purpose | Built-in Components |
|----------|---------|---------------------|
| `chunker_registry` | Document chunking strategies | hierarchical, sentence, semantic, sliding_window |
| `crawler_registry` | Web crawling backends | http, crawl4ai |
| `extractor_registry` | File content extraction | markitdown, ocr (optional) |
| `chunk_addon_registry` | Chunk enrichment add-ons | facts |

### Registering a Custom Chunker

```python
from spiderweb import chunker_registry
from spiderweb.models.document import Document, Chunk, ChunkMetadata, ChunkType

class ParagraphChunker:
    """Split documents by paragraph."""
    
    def __init__(self, min_paragraph_length: int = 50):
        self.min_length = min_paragraph_length
    
    def chunk(self, document: Document) -> list[Chunk]:
        paragraphs = document.markdown_content.split("\n\n")
        chunks = []
        
        for i, para in enumerate(paragraphs):
            if len(para) < self.min_length:
                continue
            
            chunks.append(Chunk(
                content=para.strip(),
                metadata=ChunkMetadata(
                    chunk_type=ChunkType.CUSTOM,
                    chunk_index=i,
                ),
                document_id=document.id,
            ))
        
        return chunks
    
    @classmethod
    def from_config(cls, config):
        return cls(min_paragraph_length=config.min_chunk_size)

# Register the custom chunker
chunker_registry.register("paragraph", ParagraphChunker)
```

### Using a Custom Chunker

```python
from spiderweb import Spiderweb
from spiderweb.models.config import ChunkerConfig

# Via config string
config = ChunkerConfig(strategy="paragraph")
web = Spiderweb(chunker_config=config)

# Or pass instance directly (bypasses registry)
chunker = ParagraphChunker(min_paragraph_length=100)
web = Spiderweb(chunker=chunker)
```

### Registering a Custom Crawler

```python
from spiderweb import crawler_registry
from spiderweb.crawlers.base import CrawlResult
from spiderweb.models.config import CrawlerConfig

class PlaywrightCrawler:
    """Crawler using Playwright for advanced browser automation."""
    
    async def crawl(self, url: str, config: CrawlerConfig | None = None) -> CrawlResult:
        # Your Playwright implementation
        ...
    
    async def crawl_many(
        self,
        urls: list[str],
        config: CrawlerConfig | None = None,
    ) -> list[CrawlResult]:
        results = []
        for url in urls:
            results.append(await self.crawl(url, config))
        return results

# Register
crawler_registry.register("playwright", PlaywrightCrawler)

# Use via config
config = CrawlerConfig(provider="playwright")
web = Spiderweb(crawler_config=config)
```

### Registering a Custom Extractor

```python
from spiderweb import extractor_registry
from spiderweb.models.document import Document, DocumentMetadata
from pathlib import Path

class AudioExtractor:
    """Extract text from audio files using Whisper."""
    
    def supports(self, file_path: str | Path) -> bool:
        return Path(file_path).suffix.lower() in {".mp3", ".wav", ".m4a"}
    
    async def extract(self, file_path: str | Path) -> Document:
        # Your Whisper transcription logic
        transcript = await self._transcribe(file_path)
        
        return Document(
            raw_content=transcript,
            markdown_content=transcript,
            metadata=DocumentMetadata(
                source=str(file_path),
                file_type=Path(file_path).suffix,
                extraction_method="whisper",
            ),
        )

# Register
extractor_registry.register("audio", AudioExtractor)
```

### Factory Functions

For complex initialization, use factory functions:

```python
def create_chunker_with_llm(llm_client, threshold: float = 0.7):
    """Factory that creates a semantic chunker with an LLM client."""
    from spiderweb.chunkers import SemanticChunker
    return SemanticChunker(llm_client=llm_client, threshold=threshold)

chunker_registry.register_factory("semantic-custom", create_chunker_with_llm)

# Later, create with kwargs
chunker = chunker_registry.create("semantic-custom", llm_client=my_llm, threshold=0.8)
```

### Listing Available Components

```python
from spiderweb import chunker_registry, crawler_registry

print("Available chunkers:", chunker_registry.list())
# ['hierarchical', 'semantic', 'sentence', 'sliding_window']

print("Available crawlers:", crawler_registry.list())
# ['crawl4ai', 'http']
```

### Accessing Registries via Spiderweb Class

The `Spiderweb` class provides convenient access to all registries:

```python
from spiderweb import Spiderweb

# Register via class attributes
Spiderweb.chunkers.register("my-chunker", MyChunkerClass)
Spiderweb.crawlers.register("my-crawler", MyCrawlerClass)
Spiderweb.extractors.register("my-extractor", MyExtractorClass)
```

---

## Chunk Add-Ons

Chunk add-ons run after chunking and can enrich chunks with additional data stored in `chunk.metadata.extra`. They're perfect for extracting structured information like facts, entities, or summaries that you want attached to each chunk.

### Available Add-Ons

| Add-On | Purpose | Output Location |
|--------|---------|----------------|
| `facts` | Extract factual statements from chunks | `chunk.metadata.extra["facts"]` |

### Using Built-in Add-Ons

Enable add-ons via `ChunkAddOnConfig`:

```python
from spiderweb import Spiderweb
from spiderweb.models.config import ChunkAddOnConfig
from gluellm import GlueLLM

# Enable facts extraction
config = ChunkAddOnConfig(enabled=["facts"])
web = Spiderweb(
    llm_client=GlueLLM(),
    chunk_addon_config=config,
)

# Or pass list directly
web = Spiderweb(
    llm_client=GlueLLM(),
    chunk_add_ons=["facts"],
)

# Ingest with facts extraction
result = await web.ingest("document.pdf")

# Facts are now in chunk.metadata.extra["facts"]
for chunk in result.document.chunks:
    facts = chunk.metadata.extra.get("facts", [])
    print(f"Chunk has {len(facts)} facts")
```

### Configuring Add-On Options

Some add-ons accept configuration options:

```python
from spiderweb.models.config import ChunkAddOnConfig

config = ChunkAddOnConfig(
    enabled=["facts"],
    options={
        "facts": {
            "max_facts": 10,
            "model": "gpt-4",
        }
    }
)
```

### Creating a Custom Add-On

Implement the `ChunkAddOn` protocol:

```python
from spiderweb.addons.base import ChunkAddOn
from spiderweb import chunk_addon_registry
from spiderweb.models.document import Chunk, Document

class EntityExtractionAddOn:
    """Extract named entities from chunks."""
    
    def __init__(self, llm_client=None, **kwargs):
        self.llm_client = llm_client
    
    async def process_async(
        self,
        chunks: list[Chunk],
        *,
        document: Document | None = None,
        **kwargs,
    ) -> list[Chunk]:
        """Extract entities and store in metadata.extra["entities"]."""
        for chunk in chunks:
            # Your entity extraction logic here
            entities = await self._extract_entities(chunk.content)
            chunk.metadata.extra["entities"] = entities
        return chunks
    
    async def _extract_entities(self, content: str) -> list[str]:
        # Use LLM or NER library to extract entities
        ...

# Register the add-on
chunk_addon_registry.register("entities", EntityExtractionAddOn)

# Or use a factory for complex initialization
def create_entity_addon(llm_client=None, model="gpt-4", **kwargs):
    return EntityExtractionAddOn(llm_client=llm_client, model=model)

chunk_addon_registry.register_factory("entities", create_entity_addon)
```

### Add-On Protocol

Add-ons can be synchronous or asynchronous:

```python
from spiderweb.addons.base import ChunkAddOn
from spiderweb.models.document import Chunk, Document

class MySyncAddOn:
    """Synchronous add-on."""
    
    def process(
        self,
        chunks: list[Chunk],
        *,
        document: Document | None = None,
        **kwargs,
    ) -> list[Chunk]:
        # Sync processing
        for chunk in chunks:
            chunk.metadata.extra["my_key"] = "my_value"
        return chunks

class MyAsyncAddOn:
    """Asynchronous add-on (preferred for LLM calls)."""
    
    async def process_async(
        self,
        chunks: list[Chunk],
        *,
        document: Document | None = None,
        **kwargs,
    ) -> list[Chunk]:
        # Async processing (e.g., LLM calls)
        for chunk in chunks:
            result = await self.llm_client.complete(...)
            chunk.metadata.extra["my_key"] = result
        return chunks
```

### Add-On Execution Order

Add-ons run in the order specified in `enabled`:

```python
config = ChunkAddOnConfig(enabled=["facts", "entities", "summary"])
# Runs: facts → entities → summary
```

### Accessing Add-On Output

Add-on output is stored in `chunk.metadata.extra` under add-on-defined keys:

```python
# After ingestion with facts add-on
result = await web.ingest("document.pdf")

for chunk in result.document.chunks:
    # Access facts
    facts = chunk.metadata.extra.get("facts", [])
    
    # Access other add-on outputs
    entities = chunk.metadata.extra.get("entities", [])
    summary = chunk.metadata.extra.get("summary", "")
```

### Facts Add-On Details

The built-in `facts` add-on extracts factual statements from each chunk:

- **Output**: `chunk.metadata.extra["facts"]` - list of strings
- **Options**:
  - `max_facts` (int): Maximum facts per chunk (default: 10)
  - `model` (str): LLM model to use (optional, uses default)
- **Requires**: LLM client (GlueLLM instance)

```python
from spiderweb.models.config import ChunkAddOnConfig

config = ChunkAddOnConfig(
    enabled=["facts"],
    options={"facts": {"max_facts": 5}}
)
```

---

## Pipeline Hooks

Hooks let you intercept and modify data at key pipeline stages without replacing entire components.

### Available Hook Points

| Hook Point | Trigger | Data Type |
|------------|---------|-----------|
| `BEFORE_EXTRACT` | Before file extraction | `str` (file path) |
| `AFTER_EXTRACT` | After file extraction | `Document` |
| `BEFORE_CHUNK` | Before chunking | `Document` |
| `AFTER_CHUNK` | After chunking | `list[Chunk]` |
| `BEFORE_CRAWL` | Before web crawl | `str` (URL) or `list[str]` (batch) |
| `AFTER_CRAWL` | After web crawl | `CrawlResult` or `list[CrawlResult]` (batch) |

### Basic Hook Example

```python
from spiderweb import hooks, HookPoint

def log_chunks(ctx):
    """Log chunk count after chunking."""
    chunks = ctx.data
    print(f"Created {len(chunks)} chunks")
    return ctx

hooks.register(HookPoint.AFTER_CHUNK, log_chunks)
```

### Modifying Data

Hooks can modify data by setting `ctx.modified_data`:

```python
from spiderweb import hooks, HookPoint, HookContext

def add_custom_metadata(ctx: HookContext) -> HookContext:
    """Add custom metadata to all chunks."""
    chunks = ctx.data
    
    for chunk in chunks:
        chunk.metadata.extra["processed_by"] = "my-pipeline"
        chunk.metadata.extra["version"] = "1.0"
    
    ctx.modified_data = chunks
    return ctx

hooks.register(HookPoint.AFTER_CHUNK, add_custom_metadata)
```

### Skipping Operations

Set `ctx.skip = True` to skip the operation:

```python
from pathlib import Path
from spiderweb import hooks, HookPoint, HookContext

def skip_large_files(ctx: HookContext) -> HookContext:
    """Skip extraction for files larger than 100MB."""
    file_path = Path(ctx.data)
    if file_path.stat().st_size > 100 * 1024 * 1024:
        print(f"Skipping large file: {file_path}")
        ctx.skip = True
    return ctx

hooks.register(HookPoint.BEFORE_EXTRACT, skip_large_files)
```

### Async Hooks

Hooks can be async:

```python
import aiohttp
from spiderweb import hooks, HookPoint, HookContext

async def async_validation(ctx: HookContext) -> HookContext:
    """Validate chunks with external service."""
    chunks = ctx.data
    
    async with aiohttp.ClientSession() as session:
        for chunk in chunks:
            # Call validation API
            await validate_chunk(session, chunk)
    
    return ctx

hooks.register(HookPoint.AFTER_CHUNK, async_validation)
```

### Accessing Metadata

Hooks receive metadata from the pipeline:

```python
from spiderweb import hooks, HookPoint, HookContext

def log_with_context(ctx: HookContext) -> HookContext:
    """Access metadata passed from pipeline."""
    print(f"Hook point: {ctx.hook_point}")
    print(f"Metadata: {ctx.metadata}")
    
    # Available metadata varies by hook point:
    # - AFTER_CHUNK includes: document, file_path
    # - AFTER_CRAWL includes: url, config
    
    return ctx

hooks.register(HookPoint.AFTER_CHUNK, log_with_context)
```

### Hook Manager Isolation

For testing or isolated pipelines, create a separate HookManager:

```python
from spiderweb.hooks import HookManager, HookPoint
from spiderweb.pipeline import DocumentProcessor

# Create isolated hook manager
my_hooks = HookManager()
my_hooks.register(HookPoint.BEFORE_CHUNK, my_hook)

# Pass to components
processor = DocumentProcessor(hook_manager=my_hooks)
```

### Clearing Hooks

```python
from spiderweb import hooks, HookPoint

# Clear all hooks for a specific point
hooks.clear(HookPoint.BEFORE_CHUNK)

# Clear all hooks
hooks.clear()
```

### Accessing Hooks via Spiderweb Class

```python
from spiderweb import Spiderweb, HookPoint

# Register via class attribute
Spiderweb.hooks.register(HookPoint.AFTER_CHUNK, my_callback)
```

---

## Complete Example

```python
import asyncio
from spiderweb import (
    Spiderweb,
    chunker_registry,
    hooks,
    HookPoint,
    HookContext,
    ChunkerConfig,
)
from spiderweb.models.document import Document, Chunk
from gluellm import GlueLLM


# 1. Register custom chunker
class TokenAwareChunker:
    """Chunk documents based on token count."""
    
    def __init__(self, max_tokens: int = 512):
        self.max_tokens = max_tokens
    
    def chunk(self, document: Document) -> list[Chunk]:
        # Token-based chunking logic
        ...
    
    @classmethod
    def from_config(cls, config):
        # Estimate tokens as chars / 4
        return cls(max_tokens=config.max_chunk_size // 4)


chunker_registry.register("token-aware", TokenAwareChunker)


# 2. Add logging hook
def log_pipeline(ctx: HookContext) -> HookContext:
    print(f"[{ctx.hook_point.value}] Processing...")
    return ctx


for hp in HookPoint:
    hooks.register(hp, log_pipeline)


# 3. Add chunk filtering hook
def filter_short_chunks(ctx: HookContext) -> HookContext:
    if ctx.hook_point == HookPoint.AFTER_CHUNK:
        chunks = ctx.data
        ctx.modified_data = [c for c in chunks if len(c.content) > 50]
    return ctx


hooks.register(HookPoint.AFTER_CHUNK, filter_short_chunks)


# 4. Use the customized pipeline
async def main():
    async with Spiderweb(
        llm_client=GlueLLM(),
        chunker_config=ChunkerConfig(strategy="token-aware"),
    ) as web:
        result = await web.ingest("document.pdf")
        print(f"Created {result.chunks_created} chunks")


asyncio.run(main())
```

---

## Best Practices

1. **Protocol Compliance** - Ensure custom components implement the required protocol methods:
   - Chunkers: `chunk(document: Document) -> list[Chunk]`
   - Crawlers: `crawl(url, config) -> CrawlResult` and `crawl_many(urls, config) -> list[CrawlResult]`
   - Extractors: `supports(file_path) -> bool` and `extract(file_path) -> Document`

2. **Error Handling** - Wrap hook logic in try/except to avoid breaking the pipeline:
   ```python
   def safe_hook(ctx: HookContext) -> HookContext:
       try:
           # Your logic here
           ...
       except Exception as e:
           logger.warning(f"Hook failed: {e}")
       return ctx
   ```

3. **Performance** - Keep hooks lightweight; avoid blocking I/O in sync hooks

4. **Isolation** - Use separate HookManager instances for testing:
   ```python
   def test_my_feature():
       manager = HookManager()
       manager.register(HookPoint.BEFORE_CHUNK, test_hook)
       # Test with isolated hooks
   ```

5. **Cleanup** - Call `hooks.clear()` in test teardown to avoid state leakage

6. **from_config Support** - If your custom chunker should work with `ChunkerConfig`, implement `from_config`:
   ```python
   @classmethod
   def from_config(cls, config: ChunkerConfig) -> "MyChunker":
       return cls(max_size=config.max_chunk_size)
   ```

---

## API Reference

### ComponentRegistry

```python
class ComponentRegistry[T]:
    def register(self, name: str, cls: type[T]) -> None: ...
    def register_factory(self, name: str, factory: Callable[..., T]) -> None: ...
    def get(self, name: str) -> type[T] | Callable[..., T]: ...
    def create(self, name: str, **kwargs) -> T: ...
    def list(self) -> list[str]: ...
    def __contains__(self, name: str) -> bool: ...
```

### HookManager

```python
class HookManager:
    def register(self, hook_point: HookPoint, callback: Hook) -> None: ...
    def unregister(self, hook_point: HookPoint, callback: Hook) -> None: ...
    async def run(self, hook_point: HookPoint, data: Any, **metadata) -> HookContext: ...
    def clear(self, hook_point: HookPoint | None = None) -> None: ...
    def has_hooks(self, hook_point: HookPoint) -> bool: ...
```

### HookContext

```python
@dataclass
class HookContext:
    hook_point: HookPoint      # Which hook point triggered this
    data: Any                  # The data being processed
    metadata: dict[str, Any]   # Additional context from pipeline
    skip: bool = False         # Set to True to skip the operation
    modified_data: Any = None  # Set to replace the original data
```

---

## Related Documentation

- [Crawl Feature](./CRAWL_FEATURE.md) - Web crawling and extraction
- [Local Storage](./LOCAL_STORAGE.md) - Saving crawled content locally
- [Main README](../README.md) - Getting started guide

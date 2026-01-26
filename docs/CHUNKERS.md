# Chunking Strategies

Spiderweb provides four built-in chunking strategies, each optimized for different document types and use cases.

## Strategy Comparison

| Strategy | Best For | Preserves Structure | Requires LLM | Speed |
|----------|----------|---------------------|--------------|-------|
| `hierarchical` | Markdown, documentation | Yes | No | Fast |
| `semantic` | Any text, topic-based splitting | No | Yes | Slow |
| `sentence` | Articles, prose | Partial | No | Fast |
| `sliding_window` | Any text, simple use cases | No | No | Fast |

---

## Hierarchical Chunker (Default)

Splits documents based on structure (headings, sections), preserving the document hierarchy. Creates parent-child relationships between chunks.

### When to Use

- Markdown documents with clear heading structure
- Technical documentation
- Documents where section context matters
- When you want to query by section

### How It Works

1. Parses markdown headings (`#`, `##`, `###`, etc.)
2. Creates chunks at each heading boundary
3. Splits large sections into parts if they exceed `max_chunk_size`
4. Maintains parent-child references between hierarchy levels

### Example

```python
from spiderweb import Spiderweb
from spiderweb.models.config import ChunkerConfig

web = Spiderweb(
    llm_client=llm,
    chunker_config=ChunkerConfig(
        strategy="hierarchical",
        max_chunk_size=2000,
        max_hierarchy_depth=5,
        preserve_structure=True,
    ),
)
```

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_chunk_size` | 1000 | Maximum chunk size in characters |
| `max_hierarchy_depth` | 5 | Maximum heading levels to track (1-10) |
| `preserve_structure` | True | Keep section titles in chunks |

### Input/Output Example

**Input:**
```markdown
# Introduction
This is the intro.

## Getting Started
How to begin.

### Prerequisites
What you need.
```

**Output Chunks:**
1. `# Introduction\nThis is the intro.` (parent_id: None)
2. `## Getting Started\nHow to begin.` (parent_id: chunk_1)
3. `### Prerequisites\nWhat you need.` (parent_id: chunk_2)

---

## Semantic Chunker

Splits based on semantic similarity between text segments. Uses embeddings to detect topic changes.

### When to Use

- Documents without clear structure
- When topics shift without heading markers
- Long-form text with multiple subjects
- When you need high-quality semantic boundaries

### How It Works

1. Splits text into sentences
2. Generates embeddings for each sentence
3. Calculates similarity between consecutive sentences
4. Creates boundaries when similarity drops below threshold
5. Respects min/max chunk size constraints

### Example

```python
from spiderweb import Spiderweb
from spiderweb.models.config import ChunkerConfig
from gluellm import GlueLLM

# Semantic chunking requires an LLM client for embeddings
web = Spiderweb(
    llm_client=GlueLLM(),
    chunker_config=ChunkerConfig(
        strategy="semantic",
        max_chunk_size=2000,
        min_chunk_size=200,
        semantic_threshold=0.7,  # Lower = more splits
    ),
)
```

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_chunk_size` | 1000 | Maximum chunk size in characters |
| `min_chunk_size` | 100 | Minimum chunk size in characters |
| `semantic_threshold` | 0.7 | Similarity threshold (0-1). Lower = more chunks |

### Important Notes

- **Requires async**: Use `chunk_async()` instead of `chunk()`
- **LLM dependency**: Needs a GlueLLM client for embeddings
- **Slower**: Generates embeddings for every sentence
- **Better quality**: Creates more semantically coherent chunks

---

## Sentence Chunker

Groups consecutive sentences until reaching the size limit. Never splits mid-sentence.

### When to Use

- Articles and prose
- When sentence integrity is important
- Documents with consistent paragraph structure
- When you want predictable chunk boundaries

### How It Works

1. Splits text into sentences using punctuation rules
2. Groups sentences until `max_chunk_size` is reached
3. Merges small final chunks with previous chunk if below `min_chunk_size`

### Example

```python
from spiderweb import Spiderweb
from spiderweb.models.config import ChunkerConfig

web = Spiderweb(
    llm_client=llm,
    chunker_config=ChunkerConfig(
        strategy="sentence",
        max_chunk_size=1000,
        min_chunk_size=100,
    ),
)
```

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_chunk_size` | 1000 | Maximum chunk size in characters |
| `min_chunk_size` | 100 | Minimum chunk size (small chunks merged) |

---

## Sliding Window Chunker

Fixed-size chunks with configurable overlap. The simplest and fastest approach.

### When to Use

- Any text type
- When consistency is more important than boundaries
- For quick prototyping
- When you need maximum control over chunk size

### How It Works

1. Creates chunks of exactly `max_chunk_size` characters
2. Overlaps consecutive chunks by `chunk_overlap` characters
3. Optionally respects sentence boundaries

### Example

```python
from spiderweb import Spiderweb
from spiderweb.models.config import ChunkerConfig

web = Spiderweb(
    llm_client=llm,
    chunker_config=ChunkerConfig(
        strategy="sliding_window",
        max_chunk_size=1000,
        chunk_overlap=200,
        respect_boundaries=True,  # Try to split at sentence boundaries
    ),
)
```

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_chunk_size` | 1000 | Chunk size in characters |
| `chunk_overlap` | 200 | Overlap between chunks (0-1000) |
| `respect_boundaries` | True | Split at sentence boundaries when possible |

### With vs Without Boundary Respect

**With `respect_boundaries=True`:**
- Slightly variable chunk sizes
- Complete sentences
- Better for semantic search

**With `respect_boundaries=False`:**
- Exact chunk sizes
- May cut mid-sentence
- Best for token-counted contexts

---

## Choosing a Strategy

### Decision Flowchart

```
Is your document structured with headings?
├── Yes → Use hierarchical
└── No
    └── Do topics shift without markers?
        ├── Yes → Use semantic (slower but better)
        └── No
            └── Is sentence integrity important?
                ├── Yes → Use sentence
                └── No → Use sliding_window
```

### By Document Type

| Document Type | Recommended Strategy |
|--------------|---------------------|
| Markdown/RST documentation | `hierarchical` |
| Technical specifications | `hierarchical` |
| Research papers | `semantic` |
| News articles | `sentence` |
| Legal documents | `sentence` |
| Chat logs | `sliding_window` |
| Code | `sliding_window` |
| Any (quick prototype) | `sliding_window` |

---

## CLI Usage

```bash
# Ingest with hierarchical chunking (default)
spiderweb ingest document.pdf --chunker hierarchical

# Ingest with semantic chunking
spiderweb ingest document.pdf --chunker semantic

# Ingest with sentence chunking
spiderweb ingest document.pdf --chunker sentence

# Custom chunk size
spiderweb ingest document.pdf --chunk-size 2000 --chunk-overlap 400
```

---

## Custom Chunkers

Register your own chunking strategy using the component registry.

```python
from spiderweb import chunker_registry
from spiderweb.models.document import Document, Chunk, ChunkMetadata, ChunkType


class TokenChunker:
    """Chunk by token count instead of characters."""
    
    def __init__(self, max_tokens: int = 512):
        self.max_tokens = max_tokens
    
    def chunk(self, document: Document) -> list[Chunk]:
        # Your tokenization logic
        text = document.markdown_content
        tokens = text.split()  # Simplified
        
        chunks = []
        for i in range(0, len(tokens), self.max_tokens):
            chunk_tokens = tokens[i:i + self.max_tokens]
            chunk_text = " ".join(chunk_tokens)
            
            chunks.append(Chunk(
                content=chunk_text,
                metadata=ChunkMetadata(
                    document_id=document.id,
                    chunk_index=len(chunks),
                    chunk_type=ChunkType.CUSTOM,
                ),
            ))
        
        return chunks
    
    @classmethod
    def from_config(cls, config):
        # Estimate tokens as chars / 4
        return cls(max_tokens=config.max_chunk_size // 4)


# Register the custom chunker
chunker_registry.register("token", TokenChunker)

# Use it
web = Spiderweb(
    llm_client=llm,
    chunker_config=ChunkerConfig(strategy="token"),
)
```

---

## Best Practices

1. **Start with hierarchical** for most documents
2. **Use semantic** when quality matters more than speed
3. **Match chunk size to your LLM's context window**
   - For 4K context: chunks of 500-1000 chars
   - For 8K+ context: chunks of 1000-2000 chars
4. **Overlap prevents information loss** at boundaries
5. **Test with your actual queries** to find the best strategy

---

## Related Documentation

- [Configuration Reference](./CONFIGURATION.md) - Full ChunkerConfig options
- [Extensibility Guide](./EXTENSIBILITY.md) - Registering custom chunkers
- [CLI Reference](./CLI.md) - CLI options for chunking

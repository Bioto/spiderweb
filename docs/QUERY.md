# Query Features

Spiderweb provides advanced query capabilities beyond basic vector search, including **query expansion** for improved recall and **context window retrieval** for understanding surrounding content.

## Overview

| Feature | Purpose | When to Use |
|---------|---------|-------------|
| Query Expansion | Generate multiple query variations to find more relevant results | When simple queries miss relevant content |
| Context Windows | Retrieve chunks before/after each match | When you need surrounding context for understanding |
| Semantic Guidance | Score context by relevance to a topic | When not all context is equally important |

## Query Expansion

Query expansion improves recall by generating multiple query variations and combining results using Reciprocal Rank Fusion (RRF).

### Strategies

#### Multi-Query Reformulation

Generates alternative phrasings of your query with different vocabulary and sentence structures.

```python
from spiderweb import Spiderweb
from spiderweb.models.config import QueryExpansionConfig
from gluellm import GlueLLM

async with Spiderweb(llm_client=GlueLLM()) as web:
    result = await web.query(
        "How do I deploy to production?",
        query_expansion=QueryExpansionConfig(
            enabled=True,
            strategy="multi_query",
            num_expansions=3,
            include_original=True,
        ),
    )
    
    # See what queries were generated
    print("Expanded queries:", result.expanded_queries)
    # ['How do I deploy to production?',
    #  'What is the production deployment process?',
    #  'Steps for releasing to live environment',
    #  'Production release workflow guide']
```

#### HyDE (Hypothetical Document Embeddings)

Instead of searching with the query, generates a hypothetical answer and searches for documents similar to that answer. This helps bridge the vocabulary gap between questions and answers.

```python
result = await web.query(
    "What causes memory leaks?",
    query_expansion=QueryExpansionConfig(
        enabled=True,
        strategy="hyde",
        num_expansions=1,  # Usually 1 is enough for HyDE
        include_original=True,
    ),
)

# HyDE generates a hypothetical answer like:
# "Memory leaks occur when allocated memory is not properly freed..."
```

### Reciprocal Rank Fusion (RRF)

Results from all expanded queries are combined using RRF, which:
- Gives credit to items appearing in multiple result sets
- Uses position-based scoring (higher rank = higher score)
- Doesn't require score normalization across different queries

The formula: `RRF_score = Σ (1 / (k + rank_i))`

```python
result = await web.query(
    "authentication flow",
    query_expansion=QueryExpansionConfig(
        enabled=True,
        strategy="multi_query",
        rrf_k=60,  # RRF constant (default: 60)
    ),
)

# Access RRF scores
for chunk, rrf_score in zip(result.chunks, result.rrf_scores):
    print(f"RRF Score: {rrf_score:.4f} - {chunk['content'][:50]}...")
```

### Configuration Reference

```python
class QueryExpansionConfig:
    enabled: bool = False           # Enable query expansion (opt-in)
    strategy: str = "multi_query"   # "multi_query" or "hyde"
    num_expansions: int = 3         # Number of expansions (1-10)
    include_original: bool = True   # Include original query
    rrf_k: int = 60                 # RRF constant (higher = less aggressive)
    custom_prompt: str | None       # Custom expansion prompt
```

---

## Context Window Retrieval

Retrieve chunks before and after each match to provide surrounding context.

### Basic Usage

```python
from spiderweb.models.config import ContextWindowConfig

result = await web.query(
    "Q4 revenue",
    context_window=ContextWindowConfig(
        enabled=True,
        chunks_before=2,
        chunks_after=2,
    ),
)

# Access context for each match
for idx, match_context in result.context_by_match.items():
    print(f"Match {idx}: {len(match_context.chunks)} context chunks")
    for ctx in match_context.chunks:
        print(f"  Offset {ctx.position_offset}: {ctx.chunk['content'][:50]}...")
```

### Context Modes

#### Page Mode (Default)

Retrieves surrounding pages rather than chunks. Best for documents with clear page structure.

```python
config = ContextWindowConfig(
    context_mode="page",
    chunks_before=1,  # 1 page before
    chunks_after=1,   # 1 page after
)
```

#### Chunk Mode

Retrieves surrounding chunks by chunk index. Best for documents without page structure.

```python
config = ContextWindowConfig(
    context_mode="chunk",
    chunks_before=3,
    chunks_after=3,
)
```

### Semantic Guidance

Score context chunks by relevance to a specific topic, not just proximity.

```python
config = ContextWindowConfig(
    enabled=True,
    chunks_before=5,
    chunks_after=5,
    semantic_guide="Focus on financial metrics and KPIs",
    semantic_boost_weight=0.3,
)

result = await web.query("company performance", context_window=config)

# Context chunks include semantic scores
for idx, match_context in result.context_by_match.items():
    for ctx in match_context.chunks:
        if ctx.semantic_score:
            print(f"Score {ctx.semantic_score:.3f}: {ctx.chunk['content'][:50]}...")
```

### Adaptive Expansion

Automatically expands the context window when initial results don't match the semantic guide well enough.

```python
config = ContextWindowConfig(
    enabled=True,
    chunks_before=2,
    chunks_after=2,
    semantic_guide="Look for pricing information",
    expand_on_low_score=True,       # Enable adaptive expansion
    semantic_min_score=0.5,         # Threshold to trigger expansion
    max_expansion_steps=3,          # Max expansion attempts
    expansion_step_size=2,          # Chunks added per step
)

result = await web.query("pricing", context_window=config)

# Check if expansion was used
for idx, match_context in result.context_by_match.items():
    if match_context.expansion_steps_used > 0:
        print(f"Match {idx}: Expanded {match_context.expansion_steps_used} times")
        print(f"  Final window: {match_context.final_window_size}")
```

### Configuration Reference

```python
class ContextWindowConfig:
    enabled: bool = True
    chunks_before: int = 2
    chunks_after: int = 2
    context_mode: str = "page"              # "page" or "chunk"
    semantic_guide: str | None = None       # Topic to score against
    semantic_boost_weight: float = 0.3      # Weight for semantic scoring
    deduplicate: bool = True                # Remove overlapping context
    include_match_in_context: bool = True
    
    # Adaptive expansion
    expand_on_low_score: bool = True
    semantic_min_score: float = 0.5
    max_expansion_steps: int = 3
    expansion_step_size: int = 2
```

---

## CLI Usage

### Query with Expansion

```bash
# Enable multi-query expansion
spiderweb query "machine learning" --expand --verbose

# Use HyDE strategy with custom number of expansions
spiderweb query "memory leaks" --expand --expand-strategy hyde --expand-num 5

# Custom expansion prompt
spiderweb query "API errors" --expand --expand-prompt "Generate technical variations"
```

### Query with Context

```bash
# Basic context retrieval
spiderweb query "revenue Q4" --context-before 3 --context-after 3

# Page mode with semantic guidance
spiderweb query "pricing" \
  --context-before 2 \
  --context-after 2 \
  --context-mode page \
  --semantic-guide "financial metrics and pricing tiers"
```

### Combined Example

```bash
spiderweb query "deployment process" \
  --store qdrant://localhost:6333/docs \
  --top-k 10 \
  --expand \
  --expand-strategy multi_query \
  --context-before 2 \
  --context-after 2 \
  --verbose \
  --show-scores
```

---

## Complete Example

```python
import asyncio
from gluellm import GlueLLM
from spiderweb import Spiderweb
from spiderweb.models.config import QueryExpansionConfig, ContextWindowConfig


async def advanced_query():
    async with Spiderweb(
        llm_client=GlueLLM(),
        vector_store_url="qdrant://localhost:6333/docs",
    ) as web:
        result = await web.query(
            "How does authentication work?",
            top_k=5,
            query_expansion=QueryExpansionConfig(
                enabled=True,
                strategy="multi_query",
                num_expansions=3,
            ),
            context_window=ContextWindowConfig(
                enabled=True,
                chunks_before=2,
                chunks_after=2,
                context_mode="chunk",
                semantic_guide="Focus on security and authentication flow",
            ),
        )

        print(f"Found {result.total_results} results in {result.execution_time_seconds:.2f}s")
        
        if result.expanded_queries:
            print(f"\nExpanded queries: {result.expanded_queries}")

        for idx, chunk in enumerate(result.chunks):
            print(f"\n--- Result {idx + 1} (RRF: {result.rrf_scores[idx]:.4f}) ---")
            print(chunk["content"][:200])

            # Show context
            if hasattr(result, "context_by_match") and idx in result.context_by_match:
                ctx = result.context_by_match[idx]
                print(f"  Context: {len(ctx.chunks)} chunks")


asyncio.run(advanced_query())
```

---

## Performance Considerations

1. **Query Expansion Latency**
   - Multi-query adds ~1-2 LLM calls
   - Each expanded query requires a vector search
   - Consider `num_expansions=2-3` for balance

2. **Context Retrieval**
   - Page mode typically requires fewer lookups
   - Semantic guidance adds embedding generation
   - Adaptive expansion may triple context retrieval time

3. **Best Practices**
   - Start with query expansion disabled, enable if recall is poor
   - Use `multi_query` for broad topic searches
   - Use `hyde` when query vocabulary differs from document vocabulary
   - Limit context windows to 3-5 chunks for most use cases

---

## Related Documentation

- [Crawl Feature](./CRAWL_FEATURE.md) - Web crawling and extraction
- [Configuration Reference](./CONFIGURATION.md) - All configuration options
- [CLI Reference](./CLI.md) - Complete CLI documentation

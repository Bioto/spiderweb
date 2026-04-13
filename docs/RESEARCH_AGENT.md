# Research Agent

> Persona-driven and goal-driven research workflows with parallel crawlers.

The research agent provides high-level APIs for conducting comprehensive web research using multiple parallel crawlers, then synthesizing findings into structured reports.

## Overview

The research agent combines:
- **Persona-driven research** - Provide a persona and instructions; the agent generates queries and synthesizes a report
- **Goal-driven research** - Provide a goal; the agent creates a plan, executes it, and delivers results
- **Parallel execution** - Multiple search-crawl workers run concurrently for maximum coverage
- **Intelligent synthesis** - LLM-powered report generation from aggregated research content

## Quick Start

### Persona-Driven Research

```python
from spiderweb import Spiderweb
from superglue import GlueLLM
from spiderweb.models.config import ResearchAgentConfig

async with Spiderweb(llm_client=GlueLLM()) as web:
    result = await web.research_and_report(
        persona="You are a market research analyst",
        instructions="Research competitor pricing strategies in the SaaS industry",
        research_config=ResearchAgentConfig(num_queries=5),
    )
    
    print(result.report)
    print(f"Used {len(result.queries_used)} queries")
    print(f"Crawled {sum(len(t.get_all_urls()) for t in result.traces)} URLs")
```

### Goal-Driven Research

```python
result = await web.achieve_goal(
    goal="Understand competitor X's pricing and positioning in the EU market",
    persona="You are a strategic analyst",
)

print(f"Plan: {result.plan.queries}")
print(f"Report Focus: {result.plan.report_focus}")
print(f"\nReport:\n{result.report}")
```

## Architecture

### Components

1. **Query Generation** (`spiderweb.research.agent.generate_research_queries`)
   - Takes persona + instructions
   - Uses LLM to generate diverse research queries
   - Returns list of query strings

2. **Plan Creation** (`spiderweb.research.agent.create_research_plan`)
   - Takes goal (+ optional persona/instructions)
   - Uses LLM to create structured plan
   - Returns `ResearchPlan` with queries and report_focus

3. **Parallel Execution** (`Spiderweb.research_and_report` / `achieve_goal`)
   - Runs multiple `search_crawl_extract` calls concurrently
   - Uses semaphore to cap concurrency
   - Collects all traces

4. **Aggregation** (`spiderweb.research.agent.aggregate_traces_for_report`)
   - Merges all traces into single context
   - Deduplicates by URL
   - Formats as markdown for synthesis

5. **Report Synthesis** (`spiderweb.research.agent.synthesize_report`)
   - Takes aggregated content + persona + instructions
   - Uses LLM to generate final report
   - Supports structured output via Pydantic schemas

### Data Flow

```
Persona + Instructions → Generate Queries → Parallel Search-Crawl → Aggregate → Synthesize Report
                                                      ↓
Goal → Create Plan → Execute Plan (queries + report_focus) → Synthesize Report
```

## Configuration

### ResearchAgentConfig

```python
from spiderweb.models.config import ResearchAgentConfig

config = ResearchAgentConfig(
    num_queries=5,                    # Number of queries to generate
    max_parallel_crawls=5,           # Max concurrent executions (0 = unbounded)
    max_chars_per_page_for_report=4000,  # Truncation for report context
)
```

### Integration with Other Configs

The research methods accept all the same configs as `search_crawl_extract`:

```python
from spiderweb.models.config import (
    ResearchAgentConfig,
    SearchProviderConfig,
    CrawlerConfig,
    SearchDepthConfig,
)

result = await web.research_and_report(
    persona="...",
    instructions="...",
    research_config=ResearchAgentConfig(num_queries=5),
    search_provider_config=SearchProviderConfig(provider="tavily", limit=10),
    crawler_config=CrawlerConfig(max_depth=1, wait_for_js=True),
    depth_config=SearchDepthConfig(max_search_rounds=2),
)
```

## API Reference

### Spiderweb.research_and_report()

```python
async def research_and_report(
    self,
    persona: str,
    instructions: str,
    research_config: ResearchAgentConfig | None = None,
    search_provider_config: SearchProviderConfig | None = None,
    crawler_config: CrawlerConfig | None = None,
    extraction_config: CrawlExtractionConfig | None = None,
    depth_config: SearchDepthConfig | None = None,
    report_schema: type | None = None,
    ingest: bool = False,
    save_to: str | Path | None = None,
    save_format: str | None = None,
    save_trace_to: str | Path | None = None,
    trace_format: Literal["json", "markdown", "jsonl"] | None = None,
) -> ResearchReportResult
```

**Parameters:**
- `persona` - Persona description (e.g., "You are a tech journalist")
- `instructions` - Research instructions (e.g., "Research latest AI developments")
- `research_config` - Research agent configuration
- `search_provider_config` - Search provider configuration
- `crawler_config` - Crawler configuration
- `extraction_config` - Optional extraction configuration
- `depth_config` - Multi-round search configuration
- `report_schema` - Optional Pydantic schema for structured output
- `ingest` - If True, ingest crawled content into vector store
- `save_to` - Optional directory to save crawled content
- `save_format` - Format for saved files
- `save_trace_to` - Optional path to save traces
- `trace_format` - Format for trace files

**Returns:**
- `ResearchReportResult` with `report`, `traces`, `queries_used`, and `aggregated_context_preview`

### Spiderweb.achieve_goal()

```python
async def achieve_goal(
    self,
    goal: str,
    persona: str | None = None,
    instructions: str | None = None,
    research_config: ResearchAgentConfig | None = None,
    search_provider_config: SearchProviderConfig | None = None,
    crawler_config: CrawlerConfig | None = None,
    extraction_config: CrawlExtractionConfig | None = None,
    depth_config: SearchDepthConfig | None = None,
    report_schema: type | None = None,
    ingest: bool = False,
    save_to: str | Path | None = None,
    save_format: str | None = None,
    save_trace_to: str | Path | None = None,
    trace_format: Literal["json", "markdown", "jsonl"] | None = None,
) -> GoalResult
```

**Parameters:**
- `goal` - Summary of what to achieve (e.g., "Understand competitor pricing")
- `persona` - Optional persona to guide planning and reporting
- `instructions` - Optional additional instructions
- (Other parameters same as `research_and_report`)

**Returns:**
- `GoalResult` with `plan`, `report`, `traces`, and `queries_used`

## Use Cases

### 1. Competitive Intelligence

```python
result = await web.research_and_report(
    persona="You are a competitive intelligence analyst",
    instructions="Research competitor X's product features, pricing, and market positioning",
    research_config=ResearchAgentConfig(num_queries=7),
)

# Access structured insights
print(result.report)
```

### 2. Market Research

```python
result = await web.achieve_goal(
    goal="Understand the current state of the AI safety research field",
    persona="You are a research analyst",
)

print(f"Research plan: {result.plan.queries}")
print(f"\nReport:\n{result.report}")
```

### 3. Structured Reports

```python
from pydantic import BaseModel, Field

class CompetitorAnalysis(BaseModel):
    """Structured competitor analysis report."""
    company_name: str
    pricing_tiers: list[dict]
    key_features: list[str]
    market_positioning: str
    strengths: list[str]
    weaknesses: list[str]

result = await web.research_and_report(
    persona="You are a market analyst",
    instructions="Analyze competitor X",
    report_schema=CompetitorAnalysis,
)

# Access structured data
analysis = result.report
print(f"Pricing tiers: {analysis.pricing_tiers}")
print(f"Key features: {analysis.key_features}")
```

### 4. Multi-Round Deep Research

```python
from spiderweb.models.config import SearchDepthConfig

result = await web.research_and_report(
    persona="You are a tech journalist",
    instructions="Research latest developments in quantum computing",
    depth_config=SearchDepthConfig(
        max_search_rounds=3,
        crawl_results_per_round=5,
        when_to_go_deeper="expand_queries",
    ),
)
```

## Advanced Features

### Custom Query Generation

The query generation uses LLM prompts. You can influence the queries by:
- Providing detailed persona descriptions
- Being specific in instructions
- Adjusting `num_queries` in `ResearchAgentConfig`

### Parallel Execution Control

Control concurrency to balance speed vs. resource usage:

```python
config = ResearchAgentConfig(
    max_parallel_crawls=3,  # Lower = less load, slower
)
```

Set `max_parallel_crawls=0` for unbounded concurrency (use with caution).

### Saving Traces

Each query execution produces a trace. When using `save_trace_to`, each query gets its own trace file:

```python
result = await web.research_and_report(
    persona="...",
    instructions="...",
    save_trace_to="./research_traces",
    trace_format="json",
)

# Traces saved as: ./research_traces/query1.json, query2.json, ...
```

### Ingestion Integration

Ingest crawled content for later querying:

```python
result = await web.research_and_report(
    persona="...",
    instructions="...",
    ingest=True,
)

# Later query the ingested content
query_results = await web.query("What were the key findings?")
```

## When to Use Which Method

**Use `research_and_report` when:**
- You have a clear persona and research instructions
- You want the agent to generate queries automatically
- You prefer a straightforward research → report flow

**Use `achieve_goal` when:**
- You have a high-level goal but want the agent to plan the approach
- You want transparency into the planning process
- You want the agent to determine both queries and report focus

## Performance Considerations

1. **Query Count**: More queries = more coverage but longer execution time
2. **Parallel Crawls**: Higher `max_parallel_crawls` = faster but more resource-intensive
3. **Search Rounds**: More rounds per query = deeper research but slower
4. **Report Context**: Larger `max_chars_per_page_for_report` = more context but higher LLM costs

## Troubleshooting

### LLM Returns Invalid JSON

If query generation or planning fails with JSON parsing errors:
- Check LLM client configuration
- Try a different model
- Ensure the LLM supports structured output

### Too Many Parallel Crawls

If you're hitting rate limits or resource constraints:
- Reduce `max_parallel_crawls`
- Increase `delay_between_requests` in `CrawlerConfig`
- Use fewer queries

### Report Quality Issues

If reports aren't comprehensive enough:
- Increase `num_queries` for more coverage
- Increase `max_search_rounds` in `depth_config` for deeper research
- Provide more detailed persona/instructions
- Increase `max_chars_per_page_for_report` for more context

## Related Documentation

- [Crawl Feature](./CRAWL_FEATURE.md) - Web crawling capabilities
- [Configuration](./CONFIGURATION.md) - All configuration options
- [CLI Reference](./CLI.md) - Command-line interface

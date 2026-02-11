"""Search CLI commands."""

import asyncio
from pathlib import Path

import click
from gluellm import GlueLLM
from rich.console import Console
from rich.table import Table

from spiderweb.api import Spiderweb
from spiderweb.config import settings
from spiderweb.models.config import (
    CrawlExtractionConfig,
    CrawlerConfig,
    SearchDepthConfig,
    SearchProviderConfig,
)
from spiderweb.utils.path_utils import sanitize_query_for_path

console = Console()


@click.command(name="search")
@click.argument("query", type=str)
@click.option(
    "--search-provider",
    type=str,
    default="duckduckgo",
    help="Search provider backend. Default: duckduckgo.",
)
@click.option(
    "--limit",
    type=int,
    default=10,
    help="Max search results per round. Default: 10.",
)
@click.option(
    "--crawl-provider",
    type=click.Choice(["http", "crawl4ai", "x"]),
    default="crawl4ai",
    help="Crawler backend. Default: crawl4ai.",
)
@click.option(
    "--max-rounds",
    type=int,
    default=1,
    help="Max search rounds. Default: 1.",
)
@click.option(
    "--crawl-per-round",
    type=int,
    default=3,
    help="Number of search results to crawl per round. Default: 3.",
)
@click.option(
    "--expand-queries",
    type=int,
    default=2,
    help="Number of expanded queries (for expand_queries strategy). Default: 2.",
)
@click.option(
    "--when-deeper",
    type=click.Choice(["always", "expand_queries", "if_not_found"]),
    default="always",
    help="When to run more rounds: always, expand_queries, if_not_found. Default: always.",
)
@click.option(
    "--max-pages-total",
    type=int,
    default=None,
    help="Cap total pages crawled across all rounds. Default: no limit.",
)
@click.option(
    "--crawl-relevance-prompt",
    type=str,
    default=None,
    help="Prompt for good vs bad URLs to crawl (e.g. 'Good: docs. Bad: login, ads.'). Default: none.",
)
@click.option(
    "--no-js",
    is_flag=True,
    default=False,
    help="Disable JavaScript rendering. Default: false (JS enabled).",
)
@click.option(
    "--extract",
    is_flag=True,
    default=False,
    help="Enable LLM-powered structured extraction. Default: false.",
)
@click.option(
    "--semantic-guide",
    type=str,
    default=None,
    help="Semantic guidance for extraction. Default: none.",
)
@click.option(
    "--ingest",
    is_flag=True,
    default=False,
    help="Ingest crawled content into vector store. Default: false.",
)
@click.option(
    "--store",
    type=str,
    default=None,
    help="Vector store URL (e.g. qdrant://localhost:6333/my_collection). Default: none.",
)
@click.option(
    "--save-to",
    type=click.Path(),
    default=None,
    help="Directory to save crawled content. Default: none.",
)
@click.option(
    "--save-format",
    type=click.Choice(["markdown", "html", "json", "all"]),
    default="all",
    help="Format for saved files. Default: all.",
)
@click.option(
    "--save-trace",
    type=click.Path(),
    default=None,
    help="Path to save search-crawl trace file. Default: none.",
)
@click.option(
    "--trace-format",
    type=click.Choice(["json", "markdown", "jsonl"]),
    default="json",
    help="Format for trace file. Default: json.",
)
@click.option(
    "--delay",
    type=float,
    default=1.0,
    help="Delay between requests (seconds). Default: 1.0.",
)
@click.option(
    "--timeout",
    type=int,
    default=30,
    help="Request timeout (seconds). Default: 30.",
)
@click.option(
    "--crazy",
    is_flag=True,
    default=False,
    help="CRAZY mode: no limits, runs until you press Ctrl+C. Default: false.",
)
def search_cmd(
    query: str,
    search_provider: str,
    limit: int,
    crawl_provider: str,
    max_rounds: int,
    crawl_per_round: int,
    expand_queries: int,
    when_deeper: str,
    max_pages_total: int | None,
    crawl_relevance_prompt: str | None,
    no_js: bool,
    extract: bool,
    semantic_guide: str | None,
    ingest: bool,
    store: str | None,
    save_to: str | None,
    save_format: str,
    save_trace: str | None,
    trace_format: str,
    delay: float,
    timeout: int,
    crazy: bool,
):
    """Search the web, crawl results, and optionally extract structured data.
    
    Performs a multi-round search → crawl → extract pipeline with configurable
    depth, relevance filtering, and query expansion. Builds a complete trace
    of the search session.
    
    Examples:
    
      \b
      # Basic search and crawl
      spiderweb search "python web scraping"
      
      \b
      # Search with multiple rounds and query expansion
      spiderweb search "machine learning" --max-rounds 2 --when-deeper expand_queries
      
      \b
      # Search with relevance filtering
      spiderweb search "product reviews" --crawl-relevance-prompt "Good: reviews, ratings. Bad: login, ads."
      
      \b
      # Search, crawl, and save trace
      spiderweb search "python tutorials" --save-trace ./trace.json --trace-format jsonl
    """
    asyncio.run(
        _search(
            query,
            search_provider,
            limit,
            crawl_provider,
            max_rounds,
            crawl_per_round,
            expand_queries,
            when_deeper,
            max_pages_total,
            crawl_relevance_prompt,
            no_js,
            extract,
            semantic_guide,
            ingest,
            store,
            save_to,
            save_format,
            save_trace,
            trace_format,
            delay,
            timeout,
            crazy,
        )
    )


async def _search(
    query: str,
    search_provider: str,
    limit: int,
    crawl_provider: str,
    max_rounds: int,
    crawl_per_round: int,
    expand_queries: int,
    when_deeper: str,
    max_pages_total: int | None,
    crawl_relevance_prompt: str | None,
    no_js: bool,
    extract: bool,
    semantic_guide: str | None,
    ingest: bool,
    store: str | None,
    save_to: str | None,
    save_format: str,
    save_trace: str | None,
    trace_format: str,
    delay: float,
    timeout: int,
    crazy: bool,
):
    """Async search implementation."""
    
    if crazy:
        max_rounds = 999999
        crawl_per_round = 50
        limit = 30
        max_pages_total = None
        when_deeper = "always"
    
    # Create configs
    search_config = SearchProviderConfig(
        provider=search_provider,
        limit=limit,
    )
    
    depth_config = SearchDepthConfig(
        max_search_rounds=max_rounds,
        crawl_results_per_round=crawl_per_round,
        when_to_go_deeper=when_deeper,
        num_expanded_queries=expand_queries,
        max_pages_total=max_pages_total,
    )
    
    crawler_config = CrawlerConfig(
        provider=crawl_provider,
        max_depth=1,
        max_pages=100,
        wait_for_js=not no_js,
        delay_between_requests=delay,
        timeout_seconds=timeout,
        crawl_relevance_prompt=crawl_relevance_prompt,
    )
    
    extraction_config = None
    if extract or semantic_guide:
        extraction_config = CrawlExtractionConfig(
            enabled=True,
            semantic_guide=semantic_guide,
        )
    
    # Initialize LLM client if needed
    llm = None
    if ingest or extract or crawl_relevance_prompt:
        console.print("[cyan]Initializing LLM client...[/cyan]")
        try:
            llm = GlueLLM(embedding_model=settings.embedding_model)
        except Exception as e:
            console.print(f"[red]Error: Failed to initialize LLM client: {e}[/red]")
            return
    
    # Create Spiderweb client
    web = Spiderweb(
        llm_client=llm,
        vector_store_url=store,
    )
    
    try:
        if crazy:
            console.print("[bold yellow]CRAZY mode: no limits — press Ctrl+C to stop[/bold yellow]")
        console.print(f"[cyan]Searching: {query}[/cyan]")
        console.print(f"  Search provider: {search_provider}")
        console.print(f"  Max rounds: {max_rounds}")
        console.print(f"  Crawl per round: {crawl_per_round}")
        console.print(f"  Strategy: {when_deeper}")
        
        if crawl_relevance_prompt:
            console.print(f"  Relevance filter: [green]enabled[/green]")
        
        if extraction_config and extraction_config.enabled:
            console.print(f"  Extraction: [green]enabled[/green]")
            if semantic_guide:
                console.print(f"  Guide: {semantic_guide}")
        
        with console.status("[bold cyan]Searching and crawling..."):
            trace = await web.search_crawl_extract(
                query=query,
                search_provider_config=search_config,
                crawler_config=crawler_config,
                extraction_config=extraction_config,
                depth_config=depth_config,
                ingest=ingest,
                save_to=save_to,
                save_format=save_format,
                save_trace_to=save_trace,
                trace_format=trace_format,
            )
        
        # Display results
        console.print(f"\n[green]✓[/green] Search-crawl complete")
        console.print(f"  Total rounds: {len(trace.rounds)}")
        console.print(f"  Total pages crawled: {len(trace.get_all_urls())}")
        console.print(f"  Total filtered out: {len(trace.get_all_filtered_urls())}")
        
        # Show summary table
        table = Table()
        table.add_column("Round", style="cyan")
        table.add_column("Query", style="yellow")
        table.add_column("Results", style="green")
        table.add_column("Crawled", style="magenta")
        table.add_column("Filtered", style="red")
        
        for round_data in trace.rounds:
            table.add_row(
                str(round_data.round_number),
                round_data.query[:40] + "..." if len(round_data.query) > 40 else round_data.query,
                str(len(round_data.search_results.results)),
                str(len(round_data.pages)),
                str(len(round_data.filtered_out)),
            )
        
        console.print("\n")
        console.print(table)
        
        # Show sample pages
        if trace.rounds:
            console.print("\n[bold]Sample crawled pages:[/bold]")
            for round_data in trace.rounds[:2]:  # Show first 2 rounds
                for page in round_data.pages[:3]:  # Show first 3 pages per round
                    console.print(f"  • {page.url}")
                    if page.summary:
                        console.print(f"    {page.summary[:100]}...")
        
        if save_to:
            save_dir = Path(save_to) / sanitize_query_for_path(query)
            console.print(f"\n[green]✓[/green] Crawled content saved to: {save_dir}")
        if save_trace:
            base = Path(save_trace)
            query_slug = sanitize_query_for_path(query)
            if base.suffix.lower() in (".json", ".jsonl", ".md"):
                trace_dir = base.parent
            else:
                trace_dir = base
            trace_path = trace_dir / f"{query_slug}.{trace_format}"
            console.print(f"[green]✓[/green] Trace saved to: {trace_path}")
    
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())

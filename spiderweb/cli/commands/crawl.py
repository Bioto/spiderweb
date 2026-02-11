"""Crawl CLI commands."""

import asyncio

import click
from gluellm import GlueLLM
from rich.console import Console
from rich.table import Table

from spiderweb.api import Spiderweb
from spiderweb.config import settings
from spiderweb.crawlers.base import CrawlResult
from spiderweb.models.config import CrawlExtractionConfig, CrawlerConfig

console = Console()


@click.command(name="crawl")
@click.argument("url", type=str)
@click.option(
    "--depth",
    type=int,
    default=1,
    help="Maximum crawl depth (1 = single page, >1 = follow links)",
)
@click.option(
    "--max-pages",
    type=int,
    default=10,
    help="Maximum number of pages to crawl",
)
@click.option(
    "--provider",
    type=click.Choice(["http", "crawl4ai", "x"]),
    default="crawl4ai",
    help="Crawler backend to use",
)
@click.option(
    "--no-js",
    is_flag=True,
    help="Disable JavaScript rendering (faster but may miss dynamic content)",
)
@click.option(
    "--ingest",
    is_flag=True,
    help="Ingest crawled content into vector store",
)
@click.option(
    "--store",
    type=str,
    default=None,
    help="Vector store URL (e.g., qdrant://localhost:6333/my_collection)",
)
@click.option(
    "--save-to",
    type=click.Path(),
    default=None,
    help="Save crawled content to local directory",
)
@click.option(
    "--save-format",
    type=click.Choice(["markdown", "html", "json", "all"]),
    default="all",
    help="Format for saved files (default: all)",
)
@click.option(
    "--extract",
    is_flag=True,
    help="Enable LLM-powered structured extraction",
)
@click.option(
    "--semantic-guide",
    type=str,
    default=None,
    help="Semantic guidance for extraction (e.g., 'Extract product prices and reviews')",
)
@click.option(
    "--auto-improve",
    is_flag=True,
    help="Enable auto-improvement for extraction",
)
@click.option(
    "--follow-pattern",
    type=str,
    multiple=True,
    help="Regex pattern for links to follow (can specify multiple)",
)
@click.option(
    "--exclude-pattern",
    type=str,
    multiple=True,
    help="Regex pattern for links to exclude (can specify multiple)",
)
@click.option(
    "--delay",
    type=float,
    default=1.0,
    help="Delay between requests in seconds",
)
@click.option(
    "--timeout",
    type=int,
    default=30,
    help="Request timeout in seconds",
)
def crawl_cmd(
    url: str,
    depth: int,
    max_pages: int,
    provider: str,
    no_js: bool,
    ingest: bool,
    store: str | None,
    save_to: str | None,
    save_format: str,
    extract: bool,
    semantic_guide: str | None,
    auto_improve: bool,
    follow_pattern: tuple[str, ...],
    exclude_pattern: tuple[str, ...],
    delay: float,
    timeout: int,
):
    """Crawl web content with optional structured extraction.
    
    URL can be a single URL or multiple URLs can be crawled by using --depth > 1.
    
    Examples:
    
      \b
      # Basic crawl
      spiderweb crawl https://example.com
      
      \b
      # Crawl and save to local files
      spiderweb crawl https://example.com --save-to ./crawled_data
      
      \b
      # Crawl with link following and save
      spiderweb crawl https://docs.example.com --depth 2 --max-pages 20 --save-to ./docs
      
      \b
      # Crawl and ingest to vector store
      spiderweb crawl https://example.com --ingest --store qdrant://localhost:6333/docs
      
      \b
      # Crawl, save locally AND ingest
      spiderweb crawl https://example.com --save-to ./backup --ingest --store qdrant://localhost:6333/docs
      
      \b
      # Crawl with structured extraction
      spiderweb crawl https://store.com/product --extract --semantic-guide "Extract product info"
    """
    asyncio.run(
        _crawl(
            url,
            depth,
            max_pages,
            provider,
            no_js,
            ingest,
            store,
            save_to,
            save_format,
            extract,
            semantic_guide,
            auto_improve,
            list(follow_pattern),
            list(exclude_pattern),
            delay,
            timeout,
        )
    )


async def _crawl(
    url: str,
    depth: int,
    max_pages: int,
    provider: str,
    no_js: bool,
    ingest: bool,
    store_url: str | None,
    save_to: str | None,
    save_format: str,
    extract: bool,
    semantic_guide: str | None,
    auto_improve: bool,
    follow_patterns: list[str],
    exclude_patterns: list[str],
    delay: float,
    timeout: int,
):
    """Async crawl implementation."""
    
    # Create crawler config
    crawler_config = CrawlerConfig(
        provider=provider,
        max_depth=depth,
        max_pages=max_pages,
        wait_for_js=not no_js,
        follow_patterns=follow_patterns,
        exclude_patterns=exclude_patterns,
        delay_between_requests=delay,
        timeout_seconds=timeout,
    )
    
    # Create extraction config if needed
    extraction_config = None
    if extract or semantic_guide:
        extraction_config = CrawlExtractionConfig(
            enabled=True,
            semantic_guide=semantic_guide,
            auto_improve=auto_improve,
        )
    
    # Initialize file storage if requested
    storage = None
    if save_to:
        from spiderweb.crawlers.storage import CrawlStorage
        storage = CrawlStorage(output_dir=save_to)
        console.print(f"[cyan]Saving crawled content to: {save_to}[/cyan]")
    
    # Initialize LLM client if needed
    llm = None
    if ingest or extract:
        console.print("[cyan]Initializing LLM client...[/cyan]")
        try:
            llm = GlueLLM(embedding_model=settings.embedding_model)
        except Exception as e:
            console.print(f"[red]Error: Failed to initialize LLM client: {e}[/red]")
            return
    
    # Create Spiderweb client
    web = Spiderweb(
        llm_client=llm,
        vector_store_url=store_url,
    )
    
    try:
        console.print(f"[cyan]Crawling: {url}[/cyan]")
        console.print(f"  Provider: {provider}")
        console.print(f"  Max depth: {depth}")
        console.print(f"  Max pages: {max_pages}")
        
        if extraction_config and extraction_config.enabled:
            console.print(f"  Extraction: [green]enabled[/green]")
            if semantic_guide:
                console.print(f"  Guide: {semantic_guide}")
        
        with console.status("[bold cyan]Crawling..."):
            result = await web.crawl(
                url=url,
                crawler_config=crawler_config,
                extraction_config=extraction_config,
                ingest=ingest,
            )
        
        # Save to local files if requested
        if storage and not ingest:
            # For non-ingestion mode, save the raw crawl results
            if isinstance(result, CrawlResult):
                saved_files = storage.save_crawl_result(result, format=save_format)
                console.print(f"[green]✓[/green] Saved {len(saved_files)} file(s):")
                for fmt, path in saved_files.items():
                    console.print(f"  {fmt}: {path}")
            elif isinstance(result, list):
                total_saved = 0
                for r in result:
                    if isinstance(r, CrawlResult):
                        saved_files = storage.save_crawl_result(r, format=save_format)
                        total_saved += len(saved_files)
                console.print(f"[green]✓[/green] Saved {total_saved} file(s) total")
                
                # Create index
                index_file = storage.create_index()
                console.print(f"  Index: {index_file}")
        
        # Display results based on type
        if ingest:
            # Ingestion result
            from spiderweb.models.result import BatchIngestionResult, IngestionResult
            
            if isinstance(result, IngestionResult):
                console.print(f"[green]✓[/green] Successfully crawled and ingested {url}")
                console.print(f"  Chunks created: {result.chunks_created}")
                console.print(f"  Chunks validated: {result.chunks_validated}")
            elif isinstance(result, BatchIngestionResult):
                console.print(f"[green]✓[/green] Successfully crawled and ingested {result.total_documents} pages")
                
                table = Table()
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="green")
                
                table.add_row("Total Pages", str(result.total_documents))
                table.add_row("Total Chunks", str(result.total_chunks))
                table.add_row("Chunks Validated", str(result.total_chunks_validated))
                table.add_row("Processing Time", f"{result.processing_time_seconds:.2f}s")
                
                console.print(table)
        else:
            # Crawl result(s) - CrawlResult already imported at top
            
            if isinstance(result, CrawlResult):
                console.print(f"[green]✓[/green] Successfully crawled {url}")
                console.print(f"  Status: {result.status_code}")
                console.print(f"  Content length: {len(result.content)} bytes")
                if result.markdown:
                    console.print(f"  Markdown length: {len(result.markdown)} bytes")
                console.print(f"  Links found: {len(result.links)}")
                
                # Show preview of markdown content
                if result.markdown:
                    console.print("\n[bold]Content preview:[/bold]")
                    preview = result.markdown[:500]
                    if len(result.markdown) > 500:
                        preview += "..."
                    console.print(f"[dim]{preview}[/dim]")
            
            elif isinstance(result, list):
                console.print(f"[green]✓[/green] Successfully crawled {len(result)} pages")
                
                table = Table()
                table.add_column("URL", style="cyan")
                table.add_column("Status", style="green")
                table.add_column("Content", style="yellow")
                table.add_column("Links", style="magenta")
                
                for r in result:
                    if isinstance(r, CrawlResult):
                        status_icon = "✓" if r.success else "✗"
                        status = f"{status_icon} {r.status_code}"
                        content_size = f"{len(r.content)} bytes"
                        links = str(len(r.links))
                        
                        # Truncate URL for display
                        display_url = r.url if len(r.url) <= 50 else r.url[:47] + "..."
                        
                        table.add_row(display_url, status, content_size, links)
                
                console.print(table)
    
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())


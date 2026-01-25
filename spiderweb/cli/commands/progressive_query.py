"""Progressive RAG query command."""

import asyncio
from pathlib import Path

import click
from gluellm import GlueLLM
from rich.console import Console
from rich.table import Table

from spiderweb import Spiderweb
from spiderweb.config import settings
from spiderweb.models.config import ChunkerConfig
from spiderweb.models.progressive import ProgressiveRAGConfig
from spiderweb.pipeline.progressive import ProgressiveRAGProcessor

console = Console()


@click.command(name="progressive-query")
@click.argument("query_text")
@click.option(
    "--store",
    type=str,
    help="Base vector store URL (will use _summaries and _full collections)",
)
@click.option(
    "--top-k",
    type=int,
    default=5,
    help="Number of results to return",
)
@click.option(
    "--source-file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="Path to source PDF (required for on-demand processing)",
)
@click.option(
    "--embedding-model",
    type=str,
    help="Embedding model to use (defaults to config)",
)
@click.option(
    "--use-ocr",
    is_flag=True,
    help="Use OCR for on-demand page processing",
)
def progressive_query_cmd(
    query_text: str,
    store: str | None,
    top_k: int,
    source_file: str | None,
    embedding_model: str | None,
    use_ocr: bool,
):
    """Query using Progressive RAG mode.

    Queries page summaries first, then triggers full processing if needed.

    Examples:

      \b
      # Query with automatic processing
      spiderweb progressive-query "What is the revenue?" --source-file doc.pdf

      \b
      # Query specific store
      spiderweb progressive-query "Explain the strategy" \\
          --store qdrant://localhost:6333/adobe_docs \\
          --source-file _docs/10K_2024_ADBE.pdf
    """
    asyncio.run(_progressive_query(query_text, store, top_k, source_file, embedding_model, use_ocr))


async def _progressive_query(
    query_text: str,
    store_url: str | None,
    top_k: int,
    source_file: str | None,
    embedding_model: str | None,
    use_ocr: bool,
):
    """Async implementation of progressive query."""
    console.print("[cyan]Initializing Progressive RAG...[/cyan]")

    # Create LLM client
    try:
        llm = GlueLLM(embedding_model=embedding_model or settings.embedding_model)
    except Exception as e:
        console.print(f"[red]Error: Failed to initialize LLM client: {e}[/red]")
        return

    # Parse store URLs
    base_store_url = store_url or "qdrant://localhost:6333/spiderweb"
    if "_summaries" in base_store_url or "_full" in base_store_url:
        # Remove suffix to get base
        base_store_url = base_store_url.replace("_summaries", "").replace("_full", "")

    base_parts = base_store_url.rsplit("/", 1)
    base_url = base_parts[0] if len(base_parts) > 1 else "qdrant://localhost:6333"
    collection = base_parts[1] if len(base_parts) > 1 else "spiderweb"

    summary_store_url = f"{base_url}/{collection}_summaries"
    full_store_url = f"{base_url}/{collection}_full"

    console.print(f"[dim]Summary store: {summary_store_url}[/dim]")
    console.print(f"[dim]Full store: {full_store_url}[/dim]")

    # Create Spiderweb instances
    chunker_config = ChunkerConfig()

    summary_web = Spiderweb(
        llm_client=llm,
        vector_store_url=summary_store_url,
        chunker_config=chunker_config,
    )

    full_web = Spiderweb(
        llm_client=llm,
        vector_store_url=full_store_url,
        chunker_config=chunker_config,
    )

    # Create progressive processor
    prog_config = ProgressiveRAGConfig()
    progressive_processor = ProgressiveRAGProcessor(
        llm_client=llm,
        summary_store=summary_web.document_processor.vector_store,
        full_store=full_web.document_processor.vector_store,
        document_processor=full_web.document_processor,
        config=prog_config,
        use_ocr=use_ocr,
        ocr_dpi=150,
    )

    # Execute query
    source_path = Path(source_file) if source_file else None

    console.print(f"\n[cyan]Querying:[/cyan] {query_text}\n")

    with console.status("[bold cyan]Searching..."):
        result = await progressive_processor.query(query_text, top_k, source_path)

    # Display results
    console.print(f"[green]✓[/green] Found {len(result.summary_results)} matching pages")
    console.print(f"  Query time: {result.processing_time_ms:.0f}ms")
    console.print(f"  Cache hit: {'Yes' if result.cache_hit else 'No'}")

    if result.newly_processed_pages:
        console.print(f"  [yellow]Newly processed pages: {', '.join(map(str, result.newly_processed_pages))}[/yellow]")

    # Show summary results
    if result.summary_results:
        console.print("\n[bold]Page Summaries:[/bold]\n")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Page", style="cyan", width=6)
        table.add_column("Summary", style="white")
        table.add_column("Status", style="dim", width=15)

        for ps in result.summary_results[:top_k]:
            status = "✓ Fully processed" if ps.is_fully_processed else "○ Summary only"
            summary_text = ps.summary_text[:200] + "..." if len(ps.summary_text) > 200 else ps.summary_text
            table.add_row(str(ps.page_number), summary_text, status)

        console.print(table)

    # Show full results if available
    if result.full_results:
        console.print(f"\n[bold]Full Chunks:[/bold] ({len(result.full_results)} chunks)\n")

        for i, chunk in enumerate(result.full_results[:3], 1):
            page_num = chunk.metadata.extra.get("page_number", "?")
            content = chunk.content[:300] + "..." if len(chunk.content) > 300 else chunk.content

            console.print(f"[bold cyan]{i}. Page {page_num}[/bold cyan]")
            console.print(f"{content}\n")

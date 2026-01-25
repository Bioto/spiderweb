"""Query CLI commands."""

import asyncio

import click
from gluellm import GlueLLM
from rich.console import Console
from rich.panel import Panel

from spiderweb.api import Spiderweb
from spiderweb.config import settings

console = Console()


@click.command(name="query")
@click.argument("query_text")
@click.option(
    "--store",
    type=str,
    default=None,
    help="Vector store URL (e.g., qdrant://localhost:6333/my_collection)",
)
@click.option(
    "--top-k",
    type=int,
    default=5,
    help="Number of results to return",
)
@click.option(
    "--embedding-model",
    type=str,
    default=None,
    help="Embedding model to use (defaults to config)",
)
@click.option(
    "--show-scores",
    is_flag=True,
    help="Show similarity scores",
)
def query_cmd(
    query_text: str,
    store: str | None,
    top_k: int,
    embedding_model: str | None,
    show_scores: bool,
):
    """Query the vector store.

    Examples:

      \b
      # Query the default store
      spiderweb query "What is machine learning?"

      \b
      # Query with more results
      spiderweb query "Python best practices" --top-k 10

      \b
      # Query specific Qdrant collection
      spiderweb query "How to deploy?" --store qdrant://localhost:6333/docs
    """
    asyncio.run(_query(query_text, store, top_k, embedding_model, show_scores))


async def _query(
    query_text: str,
    store_url: str | None,
    top_k: int,
    embedding_model: str | None,
    show_scores: bool,
):
    """Async query implementation."""
    # Create LLM client
    console.print("[cyan]Initializing LLM client...[/cyan]")

    try:
        llm = GlueLLM(embedding_model=embedding_model or settings.embedding_model)
    except Exception as e:
        console.print(f"[red]Error: Failed to initialize LLM client: {e}[/red]")
        return

    # Create Spiderweb client
    web = Spiderweb(llm_client=llm, vector_store_url=store_url)

    try:
        console.print(f"[cyan]Querying: {query_text}[/cyan]\n")

        with console.status("[bold cyan]Searching..."):
            result = await web.query(query_text, top_k=top_k)

        if not result.chunks:
            console.print("[yellow]No results found[/yellow]")
            return

        console.print(f"[green]Found {result.total_results} results in {result.execution_time_seconds:.3f}s[/green]\n")

        # Display results
        for idx, (chunk, score) in enumerate(zip(result.chunks, result.scores, strict=True), 1):
            # Format score
            score_str = f" (score: {score:.3f})" if show_scores else ""

            # Create panel content
            content = chunk["content"]

            # Truncate if too long
            if len(content) > 500:
                content = content[:500] + "..."

            # Add metadata
            metadata_str = f"\n[dim]Document: {chunk['document_id']} | Chunk: {chunk['chunk_index']}[/dim]"

            panel = Panel(
                content + metadata_str,
                title=f"[bold]Result {idx}{score_str}[/bold]",
                border_style="cyan",
            )

            console.print(panel)
            console.print()

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback

        console.print(traceback.format_exc())

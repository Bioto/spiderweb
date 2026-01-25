"""Query CLI commands."""

import asyncio

import click
from gluellm import GlueLLM
from rich.console import Console
from rich.panel import Panel

from spiderweb.api import Spiderweb
from spiderweb.config import settings
from spiderweb.models.config import ContextWindowConfig

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
@click.option(
    "--context-before",
    type=int,
    default=None,
    help="Number of chunks/pages before each match to include as context",
)
@click.option(
    "--context-after",
    type=int,
    default=None,
    help="Number of chunks/pages after each match to include as context",
)
@click.option(
    "--context-mode",
    type=click.Choice(["page", "chunk"]),
    default=None,
    help="Context retrieval mode (page-first or chunk-based)",
)
@click.option(
    "--semantic-guide",
    type=str,
    default=None,
    help="Prompt to guide semantic context selection",
)
def query_cmd(
    query_text: str,
    store: str | None,
    top_k: int,
    embedding_model: str | None,
    show_scores: bool,
    context_before: int | None,
    context_after: int | None,
    context_mode: str | None,
    semantic_guide: str | None,
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
      
      \b
      # Query with surrounding context
      spiderweb query "revenue Q4" --context-before 3 --context-after 3 --context-mode page
      
      \b
      # Query with semantic guidance
      spiderweb query "revenue Q4" --semantic-guide "financial metrics and KPIs"
    """
    asyncio.run(
        _query(
            query_text,
            store,
            top_k,
            embedding_model,
            show_scores,
            context_before,
            context_after,
            context_mode,
            semantic_guide,
        )
    )


async def _query(
    query_text: str,
    store_url: str | None,
    top_k: int,
    embedding_model: str | None,
    show_scores: bool,
    context_before: int | None,
    context_after: int | None,
    context_mode: str | None,
    semantic_guide: str | None,
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

    # Build context window config if any context options provided
    context_window = None
    if context_before is not None or context_after is not None or semantic_guide is not None:
        context_window = ContextWindowConfig(
            enabled=True,
            chunks_before=context_before if context_before is not None else 2,
            chunks_after=context_after if context_after is not None else 2,
            context_mode=context_mode if context_mode else "page",
            semantic_guide=semantic_guide,
        )

    try:
        console.print(f"[cyan]Querying: {query_text}[/cyan]\n")

        with console.status("[bold cyan]Searching..."):
            result = await web.query(query_text, top_k=top_k, context_window=context_window)

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

            # Add context info if available
            if context_window and hasattr(result, "context_by_match"):
                match_context = result.context_by_match.get(idx - 1)
                if match_context:
                    context_info = (
                        f"\n[dim]Context: {len(match_context.chunks)} chunks "
                        f"(window: {match_context.final_window_size[0]}/{match_context.final_window_size[1]})"
                    )
                    if match_context.expansion_steps_used > 0:
                        context_info += f", {match_context.expansion_steps_used} expansions"
                    context_info += "[/dim]"
                    metadata_str += context_info

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

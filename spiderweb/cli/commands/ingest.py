"""Ingestion CLI commands."""

import asyncio
from pathlib import Path

import click
from superglue import GlueLLM
from rich.console import Console
from rich.table import Table

from spiderweb.api import Spiderweb
from spiderweb.config import settings
from spiderweb.models.config import ChunkerConfig

console = Console()


@click.command(name="ingest")
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--chunker",
    type=click.Choice(["hierarchical", "semantic", "sentence"]),
    default="hierarchical",
    help="Chunking strategy to use",
)
@click.option(
    "--chunk-size",
    type=int,
    default=1000,
    help="Maximum chunk size in characters",
)
@click.option(
    "--chunk-overlap",
    type=int,
    default=200,
    help="Overlap between chunks in characters",
)
@click.option(
    "--store",
    type=str,
    default=None,
    help="Vector store URL (e.g., qdrant://localhost:6333/my_collection)",
)
@click.option(
    "--graph-store",
    type=str,
    default=None,
    help="Graph store URL (e.g., neo4j://user:pass@localhost:7687). If omitted, graph store is disabled.",
)
@click.option(
    "--chunk-add-on",
    "chunk_add_ons",
    type=str,
    multiple=True,
    help="Chunk add-on to enable (repeat for multiple). e.g. --chunk-add-on langextract --chunk-add-on entity_entity_relations",
)
@click.option(
    "--no-validation",
    is_flag=True,
    help="Disable chunk validation",
)
@click.option(
    "--recursive/--no-recursive",
    default=True,
    help="Process subdirectories recursively",
)
@click.option(
    "--embedding-model",
    type=str,
    default=None,
    help="Embedding model to use (defaults to config)",
)
@click.option(
    "--use-ocr",
    is_flag=True,
    help="Use OCR extraction for PDFs (slower but handles broken text encoding)",
)
@click.option(
    "--progressive",
    is_flag=True,
    help="Use Progressive RAG mode (create page summaries first, full processing on-demand)",
)
@click.option(
    "--summary-strategy",
    type=click.Choice(["first_n_chars", "llm_summary", "metadata_only"]),
    default="first_n_chars",
    help="Strategy for generating page summaries in progressive mode",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Force re-processing of all files (bypass ingest cache)",
)
def ingest_cmd(
    path: str,
    chunker: str,
    chunk_size: int,
    chunk_overlap: int,
    store: str | None,
    graph_store: str | None,
    chunk_add_ons: tuple[str, ...],
    no_validation: bool,
    recursive: bool,
    embedding_model: str | None,
    use_ocr: bool,
    progressive: bool,
    summary_strategy: str,
    force: bool,
):
    """Ingest documents into the vector store.

    PATH can be a file or directory.

    Examples:

      \b
      # Ingest a single file
      spiderweb ingest document.pdf

      \b
      # Ingest a directory with semantic chunking
      spiderweb ingest /path/to/docs --chunker semantic

      \b
      # Ingest to Qdrant
      spiderweb ingest /path/to/docs --store qdrant://localhost:6333/my_docs

      \b
      # Ingest with graph store and entity/relationship add-ons
      spiderweb ingest /path/to/docs --store qdrant://localhost:6333/docs --graph-store neo4j://localhost:7687 --chunk-add-on langextract --chunk-add-on entity_entity_relations
    """
    asyncio.run(
        _ingest(
            path,
            chunker,
            chunk_size,
            chunk_overlap,
            store,
            graph_store,
            list(chunk_add_ons),
            no_validation,
            recursive,
            embedding_model,
            use_ocr,
            progressive,
            summary_strategy,
            force,
        )
    )


async def _ingest(
    path: str,
    chunker: str,
    chunk_size: int,
    chunk_overlap: int,
    store_url: str | None,
    graph_store_url: str | None,
    chunk_add_ons: list[str],
    no_validation: bool,
    recursive: bool,
    embedding_model: str | None,
    use_ocr: bool,
    progressive: bool,
    summary_strategy: str,
    force: bool,
):
    """Async ingestion implementation."""
    path_obj = Path(path)

    # Create LLM client
    console.print("[cyan]Initializing LLM client...[/cyan]")

    try:
        llm = GlueLLM(embedding_model=embedding_model or settings.embedding_model)
    except Exception as e:
        console.print(f"[red]Error: Failed to initialize LLM client: {e}[/red]")
        return

    # Create chunker config
    from spiderweb.models.document import ChunkType

    chunker_config = ChunkerConfig(
        strategy=ChunkType(chunker),
        max_chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Progressive mode handling
    if progressive:
        from spiderweb.models.progressive import ProgressiveRAGConfig, SummaryStrategy
        from spiderweb.pipeline.progressive import ProgressiveRAGProcessor

        console.print("[yellow]Using Progressive RAG mode[/yellow]")

        # Create progressive config
        prog_config = ProgressiveRAGConfig(
            summary_strategy=SummaryStrategy(summary_strategy),
        )

        # Create summary and full stores
        # Summary store uses _summaries suffix
        summary_store_url = store_url or "qdrant://localhost:6333/spiderweb_summaries"
        if "_summaries" not in summary_store_url:
            # Add suffix if not already present
            base_url = summary_store_url.rsplit("/", 1)[0]
            collection = summary_store_url.rsplit("/", 1)[1] if "/" in summary_store_url else "spiderweb"
            summary_store_url = f"{base_url}/{collection}_summaries"
            full_store_url = f"{base_url}/{collection}_full"
        else:
            full_store_url = summary_store_url.replace("_summaries", "_full")

        # Create separate Spiderweb instances for summary and full stores
        # Note: extractor doesn't matter for progressive mode initial ingestion
        summary_web = Spiderweb(
            llm_client=llm,
            vector_store_url=summary_store_url,
            graph_store_url=graph_store_url,
            chunker_config=chunker_config,
            chunk_add_ons=chunk_add_ons or None,
        )

        full_web = Spiderweb(
            llm_client=llm,
            vector_store_url=full_store_url,
            graph_store_url=graph_store_url,
            chunker_config=chunker_config,
            chunk_add_ons=chunk_add_ons or None,
        )

        full_web.document_processor.enable_validation = not no_validation

        # Create progressive processor
        progressive_processor = ProgressiveRAGProcessor(
            llm_client=llm,
            summary_store=summary_web.document_processor.vector_store,
            full_store=full_web.document_processor.vector_store,
            document_processor=full_web.document_processor,
            config=prog_config,
            use_ocr=use_ocr,
            ocr_dpi=150,
        )

        # Process file with progressive mode
        if path_obj.is_file():
            console.print(f"[cyan]Creating page summaries for: {path_obj.name}[/cyan]")

            with console.status("[bold cyan]Generating summaries..."):
                result = await progressive_processor.ingest_with_summaries(path_obj)

            console.print(f"[green]✓[/green] Created summaries for {result['pages_summarized']} pages")
            console.print(f"  Strategy: {result['strategy']}")
            console.print(f"  Time: {result['elapsed_time_seconds']:.2f}s")
            console.print("\n[dim]Pages will be fully processed on-demand when queried.[/dim]")
        else:
            console.print("[red]Error: Progressive mode currently only supports single PDF files[/red]")

        return

    # Regular batch mode (existing logic)
    # Create extractor
    extractor = None
    if use_ocr:
        try:
            from spiderweb.extractors.ocr import OCRExtractor
            console.print("[yellow]Using OCR extraction (this will be slower)[/yellow]")
            extractor = OCRExtractor(dpi=150)
        except ImportError:
            console.print("[red]Error: OCR dependencies not installed. Install with: pip install spiderweb[ocr][/red]")
            return

    # Create Spiderweb client
    web = Spiderweb(
        llm_client=llm,
        vector_store_url=store_url,
        graph_store_url=graph_store_url,
        chunker_config=chunker_config,
        extractor=extractor,
        chunk_add_ons=chunk_add_ons or None,
    )

    web.document_processor.enable_validation = not no_validation

    try:
        if path_obj.is_file():
            # Ingest single file
            console.print(f"[cyan]Ingesting file: {path_obj.name}[/cyan]")

            with console.status("[bold cyan]Processing..."):
                result = await web.ingest(path_obj)

            if result.success:
                console.print(f"[green]✓[/green] Successfully processed {path_obj.name}")
                console.print(f"  Chunks created: {result.chunks_created}")
                console.print(f"  Chunks validated: {result.chunks_validated}")
                console.print(f"  Processing time: {result.processing_time_seconds:.2f}s")
                if result.graph_entities_written is not None:
                    console.print(f"  Graph: {result.graph_entities_written} entities, {result.graph_relationships_written or 0} relationships")
                    if result.graph_entities_written == 0 and (result.graph_relationships_written or 0) == 0:
                        console.print("  [dim]Tip: install spiderweb[langextract] and set API key so LangExtract can extract entities.[/dim]")

                if result.warnings:
                    console.print(f"  [yellow]Warnings: {len(result.warnings)}[/yellow]")
            else:
                console.print(f"[red]✗[/red] Failed to process {path_obj.name}")
                for error in result.errors:
                    console.print(f"  [red]Error: {error}[/red]")

        elif path_obj.is_dir():
            # Ingest directory
            console.print(f"[cyan]Ingesting directory: {path_obj}[/cyan]")
            console.print(f"  Recursive: {recursive}")
            console.print(f"  Chunker: {chunker}")

            # Set force flag on batch config if provided
            if force:
                web.batch_processor.config.force = True

            with console.status("[bold cyan]Processing documents..."):
                result = await web.ingest_directory(path_obj, recursive=recursive, show_progress=False)

            # Display results
            console.print("\n[bold]Ingestion Results[/bold]")

            table = Table()
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Total Documents", str(result.total_documents))
            table.add_row("Successful", str(result.successful_documents))
            table.add_row("Failed", str(result.failed_documents))
            table.add_row("Total Chunks", str(result.total_chunks))
            table.add_row("Chunks Validated", str(result.total_chunks_validated))
            table.add_row("Chunks Rejected", str(result.total_chunks_rejected))
            table.add_row("Processing Time", f"{result.processing_time_seconds:.2f}s")
            table.add_row("Avg Time/Doc", f"{result.average_time_per_document:.2f}s")

            console.print(table)

            if result.errors:
                console.print(f"\n[red]Errors encountered: {len(result.errors)}[/red]")
                for file_path, errors in result.errors.items():
                    console.print(f"  {file_path}: {errors[0]}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback

        console.print(traceback.format_exc())

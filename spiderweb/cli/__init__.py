"""Command-line interface for Spiderweb.

Provides CLI commands for document ingestion, querying, and management.
"""

import asyncio

import click

from spiderweb.cli.commands.ingest import ingest_cmd
from spiderweb.cli.commands.progressive_query import progressive_query_cmd
from spiderweb.cli.commands.query import query_cmd


@click.group()
@click.version_option(version="0.1.0", prog_name="spiderweb")
def cli():
    """Spiderweb - Document Processing and RAG Pipeline.

    Scalable document ingestion with intelligent chunking, validation,
    and vector storage for retrieval-augmented generation.
    """
    pass


# Register commands
cli.add_command(ingest_cmd)
cli.add_command(query_cmd)
cli.add_command(progressive_query_cmd)


if __name__ == "__main__":
    cli()

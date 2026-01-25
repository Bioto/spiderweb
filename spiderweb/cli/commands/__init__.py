"""CLI command package."""

from spiderweb.cli.commands.ingest import ingest_cmd
from spiderweb.cli.commands.query import query_cmd

__all__ = ["ingest_cmd", "query_cmd"]

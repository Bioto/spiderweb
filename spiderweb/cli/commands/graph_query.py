"""Graph query CLI – query Neo4j for entities and neighbors."""

import asyncio

import click
from rich.console import Console
from rich.table import Table

from spiderweb.api import parse_graph_store_url
from spiderweb.observability.logging_config import get_logger

console = Console()
logger = get_logger(__name__)

_CONNECTION_HINT = (
    "Ensure Neo4j is running and reachable. "
    "If using Docker: docker compose --profile graph up -d. "
    "If the handshake fails, set NEO4J_dbms_connector_bolt_tls_level=OPTIONAL in Neo4j."
)


@click.group(name="graph-query")
def graph_query_cmd() -> None:
    """Query the graph store (Neo4j) for entities and relationships.

    Requires a graph store URL (e.g. neo4j://user:pass@localhost:7687) and
    pip install spiderweb[neo4j]. Use after ingesting with --graph-store and
    entity/relationship add-ons.
    """


@graph_query_cmd.command(name="entities")
@click.option(
    "--graph-store",
    "graph_store_url",
    type=str,
    required=True,
    help="Graph store URL (e.g. neo4j://neo4j:pass@localhost:7687)",
)
@click.option(
    "--limit",
    type=int,
    default=50,
    help="Maximum number of entities to return",
)
@click.option(
    "--type",
    "type_filter",
    type=str,
    default=None,
    help="Filter by entity type/label (e.g. Person, Organization)",
)
def entities_cmd(
    graph_store_url: str,
    limit: int,
    type_filter: str | None,
) -> None:
    """List entities (nodes) in the graph."""
    asyncio.run(_entities(graph_store_url, limit, type_filter))


async def _entities(
    graph_store_url: str,
    limit: int,
    type_filter: str | None,
) -> None:
    try:
        config = parse_graph_store_url(graph_store_url)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        return

    try:
        from spiderweb.stores.neo4j import Neo4jGraphStore
    except ImportError:
        console.print(
            "[red]Neo4j driver not installed. Install with: pip install spiderweb[neo4j][/red]"
        )
        return

    store = Neo4jGraphStore.from_config(config)
    try:
        console.print("[cyan]Fetching entities...[/cyan]\n")
        entities = await store.list_entities(limit=limit, type_filter=type_filter)
    except Exception as e:
        console.print(f"[red]Error querying graph: {e}[/red]")
        if "Bolt" in str(e) or "connect" in str(e).lower() or "handshake" in str(e).lower():
            console.print(f"[dim]{_CONNECTION_HINT}[/dim]")
        return
    finally:
        await store.close()

    if not entities:
        console.print("[yellow]No entities found.[/yellow]")
        return

    table = Table(title=f"Entities (limit {limit})")
    table.add_column("id", style="dim")
    table.add_column("type")
    table.add_column("label")
    table.add_column("document_id", style="dim")
    for e in entities:
        table.add_row(
            str(e.get("id", ""))[:36],
            str(e.get("type", "")),
            str(e.get("label", ""))[:50],
            str(e.get("document_id") or ""),
        )
    console.print(table)
    console.print(f"\n[green]{len(entities)} entity/entities[/green]")


@graph_query_cmd.command(name="neighbors")
@click.option(
    "--graph-store",
    "graph_store_url",
    type=str,
    required=True,
    help="Graph store URL (e.g. neo4j://neo4j:pass@localhost:7687)",
)
@click.argument("entity_id", type=str)
@click.option(
    "--depth",
    type=int,
    default=1,
    help="Traversal depth (1 = direct neighbors only)",
)
def neighbors_cmd(
    graph_store_url: str,
    entity_id: str,
    depth: int,
) -> None:
    """Show relationships and neighbors for an entity by id."""
    asyncio.run(_neighbors(graph_store_url, entity_id, depth))


async def _neighbors(
    graph_store_url: str,
    entity_id: str,
    depth: int,
) -> None:
    try:
        config = parse_graph_store_url(graph_store_url)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        return

    try:
        from spiderweb.stores.neo4j import Neo4jGraphStore
    except ImportError:
        console.print(
            "[red]Neo4j driver not installed. Install with: pip install spiderweb[neo4j][/red]"
        )
        return

    store = Neo4jGraphStore.from_config(config)
    try:
        console.print(f"[cyan]Fetching neighbors for entity {entity_id!r} (depth={depth})...[/cyan]\n")
        neighbors = await store.get_neighbors(entity_id, depth=depth)
    except Exception as e:
        console.print(f"[red]Error querying graph: {e}[/red]")
        if "Bolt" in str(e) or "connect" in str(e).lower() or "handshake" in str(e).lower():
            console.print(f"[dim]{_CONNECTION_HINT}[/dim]")
        return
    finally:
        await store.close()

    if not neighbors:
        console.print("[yellow]No neighbors found for this entity.[/yellow]")
        return

    table = Table(title=f"Neighbors of {entity_id[:24]}...")
    table.add_column("source", style="dim")
    table.add_column("relation")
    table.add_column("target")
    for n in neighbors:
        table.add_row(
            str(n.get("source_label", n.get("source_id", "")))[:40],
            str(n.get("relation_type", "")),
            str(n.get("target_label", n.get("target_id", "")))[:40],
        )
    console.print(table)
    console.print(f"\n[green]{len(neighbors)} relationship(s)[/green]")

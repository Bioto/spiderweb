"""MCP server CLI command."""

import click
from rich.console import Console

console = Console()


@click.command(name="mcp")
@click.option(
    "--transport",
    type=click.Choice(["stdio", "streamable-http"]),
    default="stdio",
    help="Transport protocol. Default: stdio (for IDE integration).",
)
@click.option(
    "--host",
    type=str,
    default="127.0.0.1",
    help="Host for HTTP transport. Default: 127.0.0.1",
)
@click.option(
    "--port",
    type=int,
    default=8000,
    help="Port for HTTP transport. Default: 8000",
)
def mcp_cmd(transport: str, host: str, port: int) -> None:
    """Start the Spiderweb MCP server.

    Exposes crawl and search-crawl tools via Model Context Protocol.
    Use stdio transport for IDE integration (Cursor, Claude Desktop, etc.)
    or streamable-http for testing with MCP Inspector.

    Examples:

      \b
      # Run with stdio (for IDE integration)
      spiderweb mcp

      \b
      # Run with HTTP (for testing)
      spiderweb mcp --transport streamable-http --port 8000

    Tools exposed:
      - crawl_url: Crawl a single URL
      - crawl_urls: Crawl multiple URLs
      - search_and_crawl: Search the web and crawl results
    """
    try:
        from spiderweb.mcp_server import create_mcp_server

        mcp = create_mcp_server()

        if transport == "stdio":
            mcp.run(transport="stdio")
        elif transport == "streamable-http":
            console.print(f"[cyan]Starting MCP server (HTTP transport) on {host}:{port}...[/cyan]")
            console.print(f"[green]Connect with: http://{host}:{port}/mcp[/green]")
            mcp.run(transport="streamable-http", host=host, port=port)
        else:
            console.print(f"[red]Unknown transport: {transport}[/red]")
            raise click.Abort()

    except ImportError as e:
        console.print("[red]Error: MCP package not installed.[/red]")
        console.print("[yellow]Install with: pip install spiderweb[mcp][/yellow]")
        raise click.Abort() from e
    except Exception as e:
        console.print(f"[red]Error starting MCP server: {e}[/red]")
        raise click.Abort() from e

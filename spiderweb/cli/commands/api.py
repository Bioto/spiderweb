"""REST API server CLI command."""

import click
from rich.console import Console

console = Console()


@click.command(name="api")
@click.option(
    "--host",
    type=str,
    default="127.0.0.1",
    help="Host to bind to. Default: 127.0.0.1",
)
@click.option(
    "--port",
    type=int,
    default=8000,
    help="Port to bind to. Default: 8000",
)
@click.option(
    "--reload",
    is_flag=True,
    default=False,
    help="Enable auto-reload (for development).",
)
def api_cmd(host: str, port: int, reload: bool) -> None:
    """Start the Spiderweb REST API server.

    Exposes crawl and search-crawl endpoints via FastAPI.
    Visit http://host:port/docs for interactive API documentation.

    Examples:

      \b
      # Start server on default port
      spiderweb api

      \b
      # Start on custom host/port
      spiderweb api --host 0.0.0.0 --port 8080

      \b
      # Start with auto-reload (development)
      spiderweb api --reload

    Endpoints:
      - POST /crawl - Crawl a single URL
      - POST /crawl/batch - Crawl multiple URLs
      - POST /search - Search the web and crawl results
      - GET /health - Health check
      - GET /docs - Interactive API documentation (Swagger UI)
    """
    try:
        import uvicorn
    except ImportError as e:
        console.print("[red]Error: FastAPI/uvicorn packages not installed.[/red]")
        console.print("[yellow]Install with: pip install spiderweb[api][/yellow]")
        raise click.Abort() from e

    try:
        from spiderweb.rest_api import create_app

        app = create_app()

        console.print(f"[cyan]Starting Spiderweb API server on http://{host}:{port}[/cyan]")
        console.print(f"[green]API docs: http://{host}:{port}/docs[/green]")
        console.print(f"[green]Health check: http://{host}:{port}/health[/green]")

        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload,
        )
    except ImportError as e:
        console.print("[red]Error: FastAPI package not installed.[/red]")
        console.print("[yellow]Install with: pip install spiderweb[api][/yellow]")
        raise click.Abort() from e
    except Exception as e:
        console.print(f"[red]Error starting API server: {e}[/red]")
        raise click.Abort() from e

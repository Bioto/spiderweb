"""CLI for X search-and-expand workflow (search → top tweets → posters' followers/following)."""

import asyncio

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from spiderweb.config import settings
from spiderweb.crawlers.x import XCrawler
from spiderweb.models.config import CrawlerConfig, XScraperConfig
from spiderweb.workflows.x_search_expand import XSearchExpandWorkflow

console = Console()


@click.command(name="x-search-expand")
@click.argument("query", type=str)
@click.option(
    "--top-tweets",
    type=int,
    default=10,
    help="Number of top tweet results to use (default: 10)",
)
@click.option(
    "--max-followers",
    type=int,
    default=20,
    help="Max followers to fetch per poster (default: 20)",
)
@click.option(
    "--max-following",
    type=int,
    default=20,
    help="Max following to fetch per poster (default: 20)",
)
@click.option(
    "--graph-depth",
    type=int,
    default=1,
    help="Depth of follow graph (1 = user + lists; 2+ = recurse, default: 1)",
)
@click.option(
    "--max-users-per-level",
    type=int,
    default=10,
    help="When graph-depth > 1, max users to expand per level (default: 10)",
)
@click.option(
    "--max-tweets",
    type=int,
    default=0,
    help="Max recent tweets to pull per user (0 = off, default: 0). Shows in output and adds to CrawlResults.",
)
@click.option(
    "--bearer-token",
    type=str,
    default=None,
    envvar="SPIDERWEB_X_BEARER_TOKEN",
    help="X API Bearer token (default: SPIDERWEB_X_BEARER_TOKEN)",
)
@click.option(
    "--ingest",
    is_flag=True,
    help="Ingest results (tweets + user profiles) into vector store with entity extraction",
)
@click.option(
    "--store",
    type=str,
    default=None,
    help="Vector store URL when using --ingest (e.g. qdrant://localhost:6333)",
)
@click.option(
    "--expand-query",
    is_flag=True,
    default=False,
    help="Use GlueLLM to rewrite the search term into an X Search API query (hashtags, filters, etc.)",
)
@click.option(
    "--chunk-add-on",
    "chunk_add_ons",
    type=str,
    multiple=True,
    help="Chunk add-on to enable when using --ingest (repeat for multiple). e.g. --chunk-add-on langextract --chunk-add-on entity_entity_relations",
)
@click.option(
    "--graph-store",
    type=str,
    default=None,
    help="Graph store URL when using --ingest with entity add-ons (e.g. neo4j://localhost:7687)",
)
def x_search_expand_cmd(
    query: str,
    top_tweets: int,
    max_followers: int,
    max_following: int,
    graph_depth: int,
    max_users_per_level: int,
    max_tweets: int,
    bearer_token: str | None,
    ingest: bool,
    store: str | None,
    expand_query: bool,
    chunk_add_ons: tuple[str, ...],
    graph_store: str | None,
) -> None:
    """Run X search-and-expand: search query → top tweets → each poster's followers/following.

    Requires SPIDERWEB_X_BEARER_TOKEN or --bearer-token. This runs the same workflow
    as examples/x_search_and_follow_graph.py (search X, take top Y tweets, expand to
    posting users and their followers/following).

    Examples:

      spiderweb x-search-expand msp

      spiderweb x-search-expand "#python" --top-tweets 5 --max-followers 50

      spiderweb x-search-expand msp --ingest --store qdrant://localhost:6333

      spiderweb x-search-expand msp --ingest --store qdrant://localhost:6333 --chunk-add-on langextract --graph-store neo4j://localhost:7687
    """
    token = bearer_token or settings.x_bearer_token
    if not token:
        console.print(
            "[red]X API Bearer token required. Set SPIDERWEB_X_BEARER_TOKEN or use --bearer-token.[/red]"
        )
        raise SystemExit(1)
    asyncio.run(
        _x_search_expand(
            query=query,
            top_tweets=top_tweets,
            max_followers=max_followers,
            max_following=max_following,
            graph_depth=graph_depth,
            max_users_per_level=max_users_per_level,
            max_tweets=max_tweets,
            bearer_token=token,
            ingest=ingest,
            store_url=store,
            expand_query=expand_query,
            chunk_add_ons=list(chunk_add_ons),
            graph_store_url=graph_store,
        )
    )


async def _x_search_expand(
    query: str,
    top_tweets: int,
    max_followers: int,
    max_following: int,
    graph_depth: int,
    max_users_per_level: int,
    max_tweets: int,
    bearer_token: str,
    ingest: bool,
    store_url: str | None,
    expand_query: bool = False,
    chunk_add_ons: list[str] | None = None,
    graph_store_url: str | None = None,
) -> None:
    x_scraper = XScraperConfig(
        max_search_results=max(10, min(100, top_tweets)),
        search_max_pages=1,
        max_followers_per_user=max_followers,
        max_following_per_user=max_following,
        include_followers=True,
        include_following=True,
        graph_depth=graph_depth,
        max_users_per_level=max_users_per_level,
        delay_between_requests=0.5,
        include_tweets=max_tweets > 0,
        max_tweets_per_user=max_tweets if max_tweets > 0 else 10,
    )
    config = CrawlerConfig(
        provider="x",
        extra_config={
            "x_bearer_token": bearer_token,
            "x_scraper_config": x_scraper.model_dump(),
        },
    )
    workflow = XSearchExpandWorkflow(
        top_tweets=top_tweets,
        max_followers_per_user=max_followers,
        max_following_per_user=max_following,
        graph_depth=graph_depth,
        max_users_per_level=max_users_per_level,
    )
    progress = Progress(
        SpinnerColumn(),
        TextColumn(" [progress.description]{task.description}"),
        console=console,
    )
    task_id = progress.add_task("Starting...", total=None)
    progress.start()

    def _progress_callback(step: str, current: int | None, total: int | None, detail: str | None) -> None:
        if detail:
            progress.update(task_id, description=detail)
        elif step == "users" and current is not None and total is not None:
            progress.update(task_id, description=f"Scraping users {current}/{total}...")
        else:
            progress.update(task_id, description=step.replace("_", " ").title() + "...")

    try:
        if expand_query:
            from gluellm import GlueLLM
            from spiderweb.api import Spiderweb
            llm = GlueLLM()
            web = Spiderweb(llm_client=llm)
            result = await web.x_search_and_expand(
                query,
                top_tweets=top_tweets,
                max_followers_per_user=max_followers,
                max_following_per_user=max_following,
                graph_depth=graph_depth,
                max_users_per_level=max_users_per_level,
                crawler_config=config,
                expand_query=True,
                progress_callback=_progress_callback,
            )
            if result.resolved_query:
                console.print(f"[cyan]Expanded query:[/cyan] {result.resolved_query!r}")
            else:
                console.print(
                    f"[yellow]Using original query (expansion failed or unchanged):[/yellow] {query!r}"
                )
                if result.expansion_error:
                    console.print(f"[red]Reason:[/red] {result.expansion_error}")
        else:
            result = await workflow.run(
                query,
                crawler=XCrawler(),
                config=config,
                progress_callback=_progress_callback,
            )
    finally:
        progress.stop()

    # Top tweets (show full content in console)
    if result.tweet_results:
        console.print("\n[bold]Top tweets[/bold]")
        for i, cr in enumerate(result.tweet_results, 1):
            full_content = (cr.content or "").strip()
            console.print(f"  [dim]{i}.[/dim] [cyan]{cr.url}[/cyan]")
            for line in full_content.split("\n"):
                console.print(f"      {line}")
        console.print()

    # Summary table
    display_query = result.resolved_query if result.resolved_query else result.query
    table = Table(title=f"X search-and-expand: {display_query!r}")
    table.add_column("Tweets (top)", style="dim")
    table.add_column("Posters", style="dim")
    table.add_column("Total CrawlResults", style="dim")
    table.add_row(
        str(len(result.tweet_results)),
        ", ".join(f"@{u}" for u in result.posters) or "(none)",
        str(len(result.all_crawl_results)),
    )
    console.print(table)

    # Users expanded: show actual follower(s) and following(s) when we have them
    if result.user_results:
        user_table = Table(title="Users expanded")
        user_table.add_column("Username", style="cyan")
        user_table.add_column("Name", style="dim")
        user_table.add_column("Followers", justify="right")
        user_table.add_column("Following", justify="right")
        user_table.add_column("Tweets", justify="right")
        for username, scrape in result.user_results:
            u = scrape.user
            user_table.add_row(
                f"@{username}",
                (u.get("name") or ""),
                str(len(scrape.followers)),
                str(len(scrape.following)),
                str(len(scrape.tweets)),
            )
        console.print(user_table)

        # Show the actual 1 follower / 1 following for each poster
        console.print("\n[bold]Follower / following (first of each per poster)[/bold]")
        for username, scrape in result.user_results:
            console.print(f"  [cyan]@{username}[/cyan]")
            if scrape.followers:
                f = scrape.followers[0]
                console.print(f"    [dim]1 follower:[/dim]   @{f.get('username', f.get('id', '?'))} — {f.get('name', '')}")
            else:
                console.print("    [dim]1 follower:[/dim]   (none)")
            if scrape.following:
                f = scrape.following[0]
                console.print(f"    [dim]1 following:[/dim]  @{f.get('username', f.get('id', '?'))} — {f.get('name', '')}")
            else:
                console.print("    [dim]1 following:[/dim]  (none)")
        # Show each user's tweets when we pulled them
        for username, scrape in result.user_results:
            if scrape.tweets:
                console.print(f"\n[bold]Tweets by @{username}[/bold]")
                for i, t in enumerate(scrape.tweets, 1):
                    preview = (t.content or "").strip().split("\n")[0][:120]
                    if len((t.content or "").strip().split("\n")[0]) > 120:
                        preview += "..."
                    console.print(f"  [dim]{i}.[/dim] [cyan]{t.url}[/cyan]")
                    console.print(f"      {preview}")
        console.print()

    if ingest and result.all_crawl_results:
        from spiderweb.api import Spiderweb
        from gluellm import GlueLLM

        console.print("[cyan]Ingesting into vector store...[/cyan]")
        llm = GlueLLM()
        add_ons = chunk_add_ons or []
        async with Spiderweb(
            llm_client=llm,
            vector_store_url=store_url,
            graph_store_url=graph_store_url,
            chunk_add_ons=add_ons if add_ons else None,
        ) as web:
            batch = await web.ingest_x_crawl_results(
                result.all_crawl_results,
                crawler_name="XCrawler",
            )
        console.print(
            f"[green]Ingested: {batch.successful_documents} docs, "
            f"{batch.total_chunks} chunks[/green]"
        )
    elif ingest and not result.all_crawl_results:
        console.print("[yellow]No CrawlResults to ingest.[/yellow]")

"""Research agent CLI commands."""

import asyncio
from pathlib import Path

import click
from gluellm import GlueLLM
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from spiderweb.api import Spiderweb
from spiderweb.config import settings
from spiderweb.models.config import (
    CrawlerConfig,
    ResearchAgentConfig,
    SearchDepthConfig,
    SearchProviderConfig,
)
from spiderweb.research.agent import (
    aggregate_traces_for_report,
    create_research_plan,
    generate_research_queries,
    summarize_traces_in_batches,
    synthesize_report,
    synthesize_report_from_summaries,
)
from spiderweb.search.trace import SearchCrawlTrace

console = Console()


def _common_options(f):
    """Attach common options for research and goal."""
    f = click.option("--max-parallel", type=int, default=5, help="Max concurrent search-crawl runs. Default: 5.")(f)
    f = click.option("--search-provider", type=str, default="duckduckgo", help="Search provider. Default: duckduckgo.")(f)
    f = click.option("--limit", type=int, default=10, help="Max search results per round. Default: 10.")(f)
    f = click.option("--crawl-provider", type=click.Choice(["http", "crawl4ai"]), default="crawl4ai", help="Crawler. Default: crawl4ai.")(f)
    f = click.option("--max-rounds", type=int, default=1, help="Max search rounds per query. Default: 1.")(f)
    f = click.option("--crawl-per-round", type=int, default=3, help="Crawl this many results per round. Default: 3.")(f)
    f = click.option("--no-js", is_flag=True, default=False, help="Disable JavaScript rendering.")(f)
    f = click.option("--delay", type=float, default=1.0, help="Delay between requests (s). Default: 1.0.")(f)
    f = click.option("--timeout", type=int, default=30, help="Request timeout (s). Default: 30.")(f)
    f = click.option("--save-to", type=click.Path(), default=None, help="Directory to save crawled content.")(f)
    f = click.option("--save-format", type=click.Choice(["markdown", "html", "json", "all"]), default="all", help="Save format. Default: all.")(f)
    f = click.option("--save-trace", type=click.Path(), default=None, help="Path to save trace files.")(f)
    f = click.option("--trace-format", type=click.Choice(["json", "markdown", "jsonl"]), default="json", help="Trace format. Default: json.")(f)
    f = click.option("--ingest", is_flag=True, default=False, help="Ingest into vector store.")(f)
    f = click.option("--store", type=str, default=None, help="Vector store URL.")(f)
    f = click.option("--output", type=click.Path(), default=None, help="Save report to file.")(f)
    f = click.option("--show-full", is_flag=True, default=False, help="Show full report in console.")(f)
    f = click.option("--live-ui", is_flag=True, default=False, help="Use live terminal UI (progress/status) during crawls.")(f)
    return f


@click.command(name="research")
@click.option("--persona", required=True, type=str, help="Persona (e.g. 'You are a market research analyst').")
@click.option("--instructions", required=True, type=str, help="Research instructions.")
@click.option("--num-queries", type=int, default=5, help="Number of research queries. Default: 5.")
@_common_options
def research_cmd(
    persona: str,
    instructions: str,
    num_queries: int,
    max_parallel: int,
    search_provider: str,
    limit: int,
    crawl_provider: str,
    max_rounds: int,
    crawl_per_round: int,
    no_js: bool,
    delay: float,
    timeout: int,
    save_to: str | None,
    save_format: str,
    save_trace: str | None,
    trace_format: str,
    ingest: bool,
    store: str | None,
    output: str | None,
    show_full: bool,
    live_ui: bool,
):
    """Research a topic using persona and instructions, then synthesize a report."""
    asyncio.run(
        _research(
            persona=persona,
            instructions=instructions,
            num_queries=num_queries,
            max_parallel=max_parallel,
            search_provider=search_provider,
            limit=limit,
            crawl_provider=crawl_provider,
            max_rounds=max_rounds,
            crawl_per_round=crawl_per_round,
            no_js=no_js,
            delay=delay,
            timeout=timeout,
            save_to=save_to,
            save_format=save_format,
            save_trace=save_trace,
            trace_format=trace_format,
            ingest=ingest,
            store=store,
            output=output,
            show_full=show_full,
            live_ui=live_ui,
        )
    )


async def _research(
    persona: str,
    instructions: str,
    num_queries: int,
    max_parallel: int,
    search_provider: str,
    limit: int,
    crawl_provider: str,
    max_rounds: int,
    crawl_per_round: int,
    no_js: bool,
    delay: float,
    timeout: int,
    save_to: str | None,
    save_format: str,
    save_trace: str | None,
    trace_format: str,
    ingest: bool,
    store: str | None,
    output: str | None,
    show_full: bool,
    live_ui: bool = False,
):
    effective_parallel = max(1, min(10, max_parallel))
    research_config = ResearchAgentConfig(
        num_queries=num_queries,
        max_queries=num_queries * 2,
        max_parallel_crawls=effective_parallel,
        model=settings.model,
    )
    effective_model = research_config.model or settings.model

    search_config = SearchProviderConfig(provider=search_provider, limit=limit)
    depth_config = SearchDepthConfig(max_search_rounds=max_rounds, crawl_results_per_round=crawl_per_round)
    crawler_config = CrawlerConfig(
        provider=crawl_provider,
        max_depth=1,
        wait_for_js=not no_js,
        delay_between_requests=delay,
        timeout_seconds=timeout,
    )

    try:
        llm = GlueLLM(embedding_model=settings.embedding_model, model=effective_model)
    except Exception as e:
        console.print(f"[red]Error: Failed to initialize LLM client: {e}[/red]")
        return

    web = Spiderweb(llm_client=llm, vector_store_url=store)

    try:
        if not live_ui:
            console.print(Panel(f"[bold]Persona:[/bold] {persona}\n[bold]Instructions:[/bold] {instructions[:200]}...", title="Research Agent", border_style="cyan"))
        with console.status("[bold cyan]Generating queries..."):
            queries = await generate_research_queries(
                llm,
                persona=persona,
                instructions=instructions,
                num_queries=num_queries,
                max_queries=research_config.max_queries,
                model=effective_model,
            )
        console.print(f"[green]✓[/green] Generated {len(queries)} queries.")

        sem = asyncio.Semaphore(effective_parallel) if effective_parallel > 0 else None

        async def run_one(q: str) -> SearchCrawlTrace:
            if sem:
                async with sem:
                    return await web.search_crawl_extract(
                        query=q,
                        search_provider_config=search_config,
                        crawler_config=crawler_config,
                        depth_config=depth_config,
                        ingest=ingest,
                        save_to=save_to,
                        save_format=save_format,
                        save_trace_to=save_trace,
                        trace_format=trace_format,
                    )
            return await web.search_crawl_extract(
                query=q,
                search_provider_config=search_config,
                crawler_config=crawler_config,
                depth_config=depth_config,
                ingest=ingest,
                save_to=save_to,
                save_format=save_format,
                save_trace_to=save_trace,
                trace_format=trace_format,
            )

        if live_ui:
            from rich.live import Live
            from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
            progress = Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            )
            task = progress.add_task("Parallel search-crawl...", total=len(queries))
            n_done = 0

            async def run_one_tracked(q: str) -> SearchCrawlTrace:
                result = await run_one(q)
                nonlocal n_done
                n_done += 1
                progress.update(task, completed=n_done)
                return result

            with Live(progress, console=console, refresh_per_second=4):
                traces = await asyncio.gather(*[run_one_tracked(q) for q in queries])
        else:
            with console.status("[bold cyan]Running parallel search-crawl..."):
                traces = await asyncio.gather(*[run_one(q) for q in queries])

        all_traces = [t for t in traces if isinstance(t, SearchCrawlTrace)]
        if not all_traces:
            console.print("[red]No successful crawls. Cannot generate report.[/red]")
            return

        if research_config.use_batched_summarization:
            with console.status("[bold cyan]Summarizing in batches..."):
                batch_summaries = await summarize_traces_in_batches(
                    llm,
                    persona=persona,
                    instructions=instructions,
                    traces=all_traces,
                    store=None,
                    batch_size_pages=research_config.summary_batch_size_pages,
                    max_chars_per_page=research_config.max_chars_per_page_for_report,
                    model=effective_model,
                )
            with console.status("[bold cyan]Generating report..."):
                report = await synthesize_report_from_summaries(
                    llm,
                    persona=persona,
                    instructions=instructions,
                    batch_summaries=batch_summaries,
                    report_focus=None,
                    model=effective_model,
                )
        else:
            with console.status("[bold cyan]Aggregating content..."):
                aggregated = await aggregate_traces_for_report(
                    all_traces,
                    max_chars_per_page=research_config.max_chars_per_page_for_report,
                )
            with console.status("[bold cyan]Generating report..."):
                report = await synthesize_report(
                    llm,
                    persona=persona,
                    instructions=instructions,
                    aggregated_content=aggregated,
                    report_focus=None,
                    model=effective_model,
                )

        report_text = report if isinstance(report, str) else str(report)
        if output:
            Path(output).write_text(report_text, encoding="utf-8")
            console.print(f"[green]✓[/green] Report saved to [bold]{output}[/bold]")
        if show_full or not output:
            if show_full:
                console.print(report_text)
            else:
                preview = report_text[:5000] + ("..." if len(report_text) > 5000 else "")
                console.print(Panel(preview, title="Report", border_style="blue"))
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise


@click.command(name="goal")
@click.argument("goal", type=str)
@click.option("--persona", type=str, default=None, help="Optional persona.")
@click.option("--instructions", type=str, default=None, help="Optional instructions.")
@_common_options
def goal_cmd(
    goal: str,
    persona: str | None,
    instructions: str | None,
    max_parallel: int,
    search_provider: str,
    limit: int,
    crawl_provider: str,
    max_rounds: int,
    crawl_per_round: int,
    no_js: bool,
    delay: float,
    timeout: int,
    save_to: str | None,
    save_format: str,
    save_trace: str | None,
    trace_format: str,
    ingest: bool,
    store: str | None,
    output: str | None,
    show_full: bool,
    live_ui: bool,
):
    """Achieve a goal by planning and executing research, then synthesize a report."""
    asyncio.run(
        _goal(
            goal=goal,
            persona=persona,
            instructions=instructions,
            max_parallel=max_parallel,
            search_provider=search_provider,
            limit=limit,
            crawl_provider=crawl_provider,
            max_rounds=max_rounds,
            crawl_per_round=crawl_per_round,
            no_js=no_js,
            delay=delay,
            timeout=timeout,
            save_to=save_to,
            save_format=save_format,
            save_trace=save_trace,
            trace_format=trace_format,
            ingest=ingest,
            store=store,
            output=output,
            show_full=show_full,
            live_ui=live_ui,
        )
    )


async def _goal(
    goal: str,
    persona: str | None,
    instructions: str | None,
    max_parallel: int,
    search_provider: str,
    limit: int,
    crawl_provider: str,
    max_rounds: int,
    crawl_per_round: int,
    no_js: bool,
    delay: float,
    timeout: int,
    save_to: str | None,
    save_format: str,
    save_trace: str | None,
    trace_format: str,
    ingest: bool,
    store: str | None,
    output: str | None,
    show_full: bool,
    live_ui: bool = False,
):
    effective_parallel = max(1, min(10, max_parallel))
    research_config = ResearchAgentConfig(max_parallel_crawls=effective_parallel, model=settings.model)
    effective_model = research_config.model or settings.model

    search_config = SearchProviderConfig(provider=search_provider, limit=limit)
    depth_config = SearchDepthConfig(max_search_rounds=max_rounds, crawl_results_per_round=crawl_per_round)
    crawler_config = CrawlerConfig(
        provider=crawl_provider,
        max_depth=1,
        wait_for_js=not no_js,
        delay_between_requests=delay,
        timeout_seconds=timeout,
    )

    try:
        llm = GlueLLM(embedding_model=settings.embedding_model, model=effective_model)
    except Exception as e:
        console.print(f"[red]Error: Failed to initialize LLM client: {e}[/red]")
        return

    web = Spiderweb(llm_client=llm, vector_store_url=store)

    try:
        console.print(Panel(f"[bold]Goal:[/bold] {goal}", title="Goal-Driven Research", border_style="cyan"))
        with console.status("[bold cyan]Creating plan..."):
            plan = await create_research_plan(
                llm,
                goal=goal,
                persona=persona,
                instructions=instructions,
                model=effective_model,
            )
        console.print(f"[green]✓[/green] Plan: {len(plan.queries)} queries. Focus: {plan.report_focus[:80]}...")

        sem = asyncio.Semaphore(effective_parallel) if effective_parallel > 0 else None

        async def run_one(q: str) -> SearchCrawlTrace:
            if sem:
                async with sem:
                    return await web.search_crawl_extract(
                        query=q,
                        search_provider_config=search_config,
                        crawler_config=crawler_config,
                        depth_config=depth_config,
                        ingest=ingest,
                        save_to=save_to,
                        save_format=save_format,
                        save_trace_to=save_trace,
                        trace_format=trace_format,
                    )
            return await web.search_crawl_extract(
                query=q,
                search_provider_config=search_config,
                crawler_config=crawler_config,
                depth_config=depth_config,
                ingest=ingest,
                save_to=save_to,
                save_format=save_format,
                save_trace_to=save_trace,
                trace_format=trace_format,
            )

        if live_ui:
            from rich.live import Live
            from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
            progress = Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            )
            task = progress.add_task("Executing plan...", total=len(plan.queries))
            n_done = 0

            async def run_one_tracked(q: str) -> SearchCrawlTrace:
                result = await run_one(q)
                nonlocal n_done
                n_done += 1
                progress.update(task, completed=n_done)
                return result

            with Live(progress, console=console, refresh_per_second=4):
                traces = await asyncio.gather(*[run_one_tracked(q) for q in plan.queries])
        else:
            with console.status("[bold cyan]Executing plan..."):
                traces = await asyncio.gather(*[run_one(q) for q in plan.queries])

        all_traces = [t for t in traces if isinstance(t, SearchCrawlTrace)]
        if not all_traces:
            console.print("[red]No successful crawls. Cannot generate report.[/red]")
            return

        if research_config.use_batched_summarization:
            with console.status("[bold cyan]Summarizing in batches..."):
                batch_summaries = await summarize_traces_in_batches(
                    llm,
                    persona=persona or "You are a research agent",
                    instructions=instructions or goal,
                    traces=all_traces,
                    store=None,
                    batch_size_pages=research_config.summary_batch_size_pages,
                    max_chars_per_page=research_config.max_chars_per_page_for_report,
                    model=effective_model,
                )
            with console.status("[bold cyan]Generating report..."):
                report = await synthesize_report_from_summaries(
                    llm,
                    persona=persona or "You are a research agent",
                    instructions=instructions or goal,
                    batch_summaries=batch_summaries,
                    report_focus=plan.report_focus,
                    model=effective_model,
                )
        else:
            with console.status("[bold cyan]Aggregating content..."):
                aggregated = await aggregate_traces_for_report(
                    all_traces,
                    max_chars_per_page=research_config.max_chars_per_page_for_report,
                )
            with console.status("[bold cyan]Generating report..."):
                report = await synthesize_report(
                    llm,
                    persona=persona or "You are a research agent",
                    instructions=instructions or goal,
                    aggregated_content=aggregated,
                    report_focus=plan.report_focus,
                    model=effective_model,
                )

        report_text = report if isinstance(report, str) else str(report)
        if output:
            Path(output).write_text(report_text, encoding="utf-8")
            console.print(f"[green]✓[/green] Report saved to [bold]{output}[/bold]")
        if show_full or not output:
            if show_full:
                console.print(report_text)
            else:
                preview = report_text[:5000] + ("..." if len(report_text) > 5000 else "")
                console.print(Panel(preview, title="Report", border_style="blue"))
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise

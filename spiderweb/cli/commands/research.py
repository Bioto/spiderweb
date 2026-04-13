"""Research agent CLI commands."""

import asyncio
from pathlib import Path
from typing import Any

import click
from superglue import GlueLLM
from rich.console import Console
from rich.panel import Panel

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
    format_listings_as_list,
    gather_listings_from_traces,
    generate_research_queries,
    summarize_traces_in_batches,
    synthesize_report,
    synthesize_report_from_listings_and_summaries,
    synthesize_report_from_summaries,
    validate_items,
)
from spiderweb.research.models import PipelineType, RoundReflection
from spiderweb.search.content_filter import filter_stale_listings
from spiderweb.search.trace import SearchCrawlTrace
from spiderweb.workflows.research import GoalResearchWorkflow

console = Console()


def _unique_crawled_page_count(traces: list[SearchCrawlTrace]) -> int:
    """Count distinct page URLs across traces (for extraction diagnostics)."""
    urls: set[str] = set()
    for t in traces:
        for r in t.rounds:
            for p in r.pages:
                if p.url:
                    urls.add(p.url)
    return len(urls)


def _common_options(f):
    """Attach common options for research and goal."""
    f = click.option("--max-parallel", type=int, default=10, help="Max concurrent search-crawl runs. Default: 10.")(f)
    f = click.option("--search-provider", type=str, default="duckduckgo", help="Search provider. Default: duckduckgo.")(f)
    f = click.option(
        "--recency",
        type=click.Choice(["d", "w", "m", "y"]),
        default=None,
        help="Filter results by recency: d (day), w (week), m (month), y (year). Maps to timelimit (duckduckgo), tbs (firecrawl), time_range (searxng).",
    )(f)
    f = click.option(
        "--location",
        type=str,
        default=None,
        help="Geo-target search results (firecrawl only). E.g. 'San Francisco,California,United States'. Default: none.",
    )(f)
    f = click.option("--limit", type=int, default=10, help="Max search results per round. Default: 10.")(f)
    f = click.option(
        "--crawl-provider",
        type=click.Choice(["http", "crawl4ai", "firecrawl", "x"]),
        default="crawl4ai",
        help="Crawler. Default: crawl4ai.",
    )(f)
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
    recency: str | None,
    location: str | None,
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
            recency=recency,
            location=location,
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
    recency: str | None,
    location: str | None,
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

    recency_to_tbs = {"d": "qdr:d", "w": "qdr:w", "m": "qdr:m", "y": "qdr:y"}
    recency_to_searx_time_range = {"d": "day", "w": "week", "m": "month", "y": "year"}
    if recency:
        if search_provider == "firecrawl":
            search_extra: dict = {"tbs": recency_to_tbs[recency]}
        elif search_provider == "searxng":
            search_extra = {"time_range": recency_to_searx_time_range[recency]}
        else:
            search_extra = {"timelimit": recency}
    else:
        search_extra = {}
    if location and search_provider == "firecrawl":
        search_extra["location"] = location

    search_config = SearchProviderConfig(
        provider=search_provider,
        limit=limit,
        extra_config=search_extra,
    )
    depth_config = SearchDepthConfig(max_search_rounds=max_rounds, crawl_results_per_round=crawl_per_round)
    crawler_config = CrawlerConfig(
        provider=crawl_provider,
        max_depth=1,
        wait_for_js=not no_js,
        delay_between_requests=delay,
        timeout_seconds=timeout,
        # domcontentloaded: forum/listing pages often never fire full "load" (ads/trackers).
        # scan_full_page still scrolls afterward for lazy-loaded content.
        wait_until="domcontentloaded",
        scan_full_page=True,
        scroll_delay=0.5,
        word_count_threshold=10,
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
        if recency:
            console.print(f"[dim]Search recency filter: {recency} (d=day, w=week, m=month, y=year)[/dim]")
        with console.status("[bold cyan]Generating queries..."):
            queries = await generate_research_queries(
                llm,
                persona=persona,
                instructions=instructions,
                num_queries=num_queries,
                max_queries=research_config.max_queries,
                model=effective_model,
                search_provider=search_provider,
            )
        console.print(f"[green]✓[/green] Generated {len(queries)} queries.")

        sem = asyncio.Semaphore(effective_parallel) if effective_parallel > 0 else None

        async def run_one(q: str) -> SearchCrawlTrace:
            try:
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
            except Exception as e:
                console.print(f"[yellow]⚠ Query failed (will continue): {q!r} — {e}[/yellow]")
                return SearchCrawlTrace(original_query=q)

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
@click.option("--target", type=int, default=None, help="Target number of listings to find (iterative search).")
@click.option(
    "--search-rounds",
    type=click.IntRange(1, 150),
    default=10,
    help="Max iterative search rounds when using --target. Default: 10.",
)
@_common_options
def goal_cmd(
    goal: str,
    persona: str | None,
    instructions: str | None,
    target: int | None,
    search_rounds: int,
    max_parallel: int,
    search_provider: str,
    recency: str | None,
    location: str | None,
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
            target=target,
            search_rounds=search_rounds,
            max_parallel=max_parallel,
            search_provider=search_provider,
            recency=recency,
            location=location,
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
    target: int | None,
    search_rounds: int,
    max_parallel: int,
    search_provider: str,
    recency: str | None,
    location: str | None,
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
        max_parallel_crawls=effective_parallel,
        model=settings.model,
        target_listings=target,
        max_search_rounds=search_rounds,
    )
    effective_model = research_config.model or settings.model

    recency_to_tbs = {"d": "qdr:d", "w": "qdr:w", "m": "qdr:m", "y": "qdr:y"}
    recency_to_searx_time_range = {"d": "day", "w": "week", "m": "month", "y": "year"}
    if recency:
        if search_provider == "firecrawl":
            search_extra: dict = {"tbs": recency_to_tbs[recency]}
        elif search_provider == "searxng":
            search_extra = {"time_range": recency_to_searx_time_range[recency]}
        else:
            search_extra = {"timelimit": recency}
    else:
        search_extra = {}
    if location and search_provider == "firecrawl":
        search_extra["location"] = location

    search_config = SearchProviderConfig(
        provider=search_provider,
        limit=limit,
        extra_config=search_extra,
    )
    depth_config = SearchDepthConfig(max_search_rounds=max_rounds, crawl_results_per_round=crawl_per_round)
    crawler_config = CrawlerConfig(
        provider=crawl_provider,
        max_depth=1,
        wait_for_js=not no_js,
        delay_between_requests=delay,
        timeout_seconds=timeout,
        # domcontentloaded: forum/listing pages often never fire full "load" (ads/trackers).
        wait_until="domcontentloaded",
        scan_full_page=True,
        scroll_delay=0.5,
        word_count_threshold=10,
    )

    try:
        llm = GlueLLM(embedding_model=settings.embedding_model, model=effective_model)
    except Exception as e:
        console.print(f"[red]Error: Failed to initialize LLM client: {e}[/red]")
        return

    web = Spiderweb(llm_client=llm, vector_store_url=store)

    try:
        console.print(Panel(f"[bold]Goal:[/bold] {goal}", title="Goal-Driven Research", border_style="cyan"))

        if recency:
            console.print(f"[dim]Search recency filter: {recency} (d=day, w=week, m=month, y=year)[/dim]")
        with console.status("[bold cyan]Creating plan..."):
            plan = await create_research_plan(
                llm,
                goal=goal,
                persona=persona,
                instructions=instructions,
                model=effective_model,
                search_provider=search_provider,
            )
        console.print(f"[green]✓[/green] Plan: {len(plan.queries)} queries. Focus: {plan.report_focus[:80]}...")
        if plan.search_strategy.rationale.strip():
            console.print(f"[dim]Search strategy: {plan.search_strategy.rationale}[/dim]")

        sem = asyncio.Semaphore(effective_parallel) if effective_parallel > 0 else None

        async def run_one(q: str) -> SearchCrawlTrace:
            try:
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
                            search_strategy=plan.search_strategy,
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
                    search_strategy=plan.search_strategy,
                )
            except Exception as e:
                console.print(f"[yellow]⚠ Query failed (will continue): {q!r} — {e}[/yellow]")
                return SearchCrawlTrace(original_query=q)

        listing_style_pipeline = plan.pipeline in (PipelineType.listing, PipelineType.listing_report)
        final_listings: list = []

        # Iterative search when target is set and listing-style pipeline is active
        if target is not None and listing_style_pipeline:
            console.print(f"[dim]Iterative search mode: targeting {target} listings[/dim]")
            if plan.acceptance_criteria is None:
                console.print(
                    "[dim]Note: listing/listing_report pipeline without acceptance_criteria; "
                    "items are not LLM-validated against goal rules.[/dim]"
                )

            def research_progress(
                step: str, current: int | None, total: int | None, detail: Any | None
            ) -> None:
                if step == "reflect_start":
                    console.print(f"[dim]Round {current}: Reflecting to plan next queries...[/dim]")
                elif step == "reflection" and isinstance(detail, RoundReflection):
                    ref = detail
                    lines = [f"• {q}" for q in ref.queries[:12]]
                    if len(ref.queries) > 12:
                        lines.append(f"… (+{len(ref.queries) - 12} more)")
                    q_block = "\n".join(lines) if lines else "(none)"
                    body = (
                        f"[bold]Found[/bold]: {ref.what_we_found}\n\n"
                        f"[bold]Missing[/bold]: {ref.what_is_missing}\n\n"
                        f"[bold]Strategy[/bold]: {ref.search_adjustment}\n\n"
                        f"[dim]{ref.reasoning}[/dim]\n\n"
                        f"[bold]New queries[/bold]:\n{q_block}"
                    )
                    console.print(
                        Panel(
                            body,
                            title=f"[cyan]Round {current} reflection[/cyan]",
                            border_style="cyan",
                        )
                    )
                elif step == "round_queries":
                    nq = total if total is not None else 0
                    console.print(f"[bold cyan]Round {current}:[/bold cyan] Running {nq} queries...")
                elif step == "round_done" and isinstance(detail, dict):
                    extracted_count = int(detail.get("extracted_count", 0))
                    pages_crawled = int(detail.get("pages_crawled", 0))
                    stale_dropped = int(detail.get("stale_dropped", 0))
                    after_stale = int(detail.get("after_stale_count", 0))
                    deduped_new = int(detail.get("deduped_new", 0))
                    total_listings = int(detail.get("total_listings", 0))
                    dup_skipped = int(detail.get("dup_skipped", 0))
                    r = int(current or 0)
                    console.print(
                        f"[dim]  Extracted {extracted_count} listing(s) from "
                        f"{pages_crawled} unique page(s) this round[/dim]"
                    )
                    if stale_dropped:
                        console.print(
                            f"[dim]  Filtered {stale_dropped} stale listing(s); "
                            f"{after_stale} kept[/dim]"
                        )
                    verdicts = detail.get("verdicts")
                    if verdicts:
                        for v in verdicts:
                            if not v.passed:
                                console.print(
                                    f"[dim]  FAIL: {v.item.url} — {'; '.join(v.rule_results)}[/dim]"
                                )
                            elif v.unknown_count:
                                console.print(
                                    f"[dim]  PASS (partial): {v.item.url} — "
                                    f"{v.unknown_count} rule(s) unverifiable[/dim]"
                                )
                        fail_n = sum(1 for x in verdicts if not x.passed)
                        if fail_n:
                            console.print(f"[dim]  Rejected {fail_n} item(s) by acceptance criteria[/dim]")
                    if dup_skipped:
                        console.print(f"[dim]  Skipped {dup_skipped} duplicate(s) already in prior rounds[/dim]")
                    console.print(
                        f"[green]✓[/green] Round {r}: Found {deduped_new} new listings ({total_listings} total)"
                    )

            wf = GoalResearchWorkflow(research_config=research_config)
            listing_result = await wf.run_iterative_listings(
                goal=goal,
                plan=plan,
                llm_client=llm,
                run_query=run_one,
                target_listings=target,
                research_config=research_config,
                model=effective_model,
                persona=persona or "You are a research agent",
                progress_callback=research_progress,
            )
            all_listings = listing_result.listings
            all_traces = listing_result.traces
            report = listing_result.report
            final_listings = all_listings

            if len(all_listings) >= target:
                console.print(f"[green]✓[/green] Target of {target} listings reached!")

            if not all_listings and plan.pipeline != PipelineType.listing_report:
                console.print("[red]No listings found. Cannot generate report.[/red]")
                return
            if plan.pipeline == PipelineType.listing_report and not all_traces:
                console.print("[red]No successful crawls. Cannot generate report.[/red]")
                return
            if plan.pipeline == PipelineType.listing_report:
                if not all_listings:
                    console.print(
                        "[yellow]No items extracted; synthesizing report from crawl summaries only.[/yellow]"
                    )
                with console.status("[bold cyan]Summarizing crawls for report..."):
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
                with console.status("[bold cyan]Generating report from items and summaries..."):
                    report = await synthesize_report_from_listings_and_summaries(
                        llm,
                        persona or "You are a research agent",
                        instructions or goal,
                        all_listings,
                        batch_summaries,
                        report_focus=plan.report_focus,
                        report_format_instructions=plan.report_format_instructions,
                        model=effective_model,
                    )
        else:
            # One-shot execution (original behavior)
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

            if listing_style_pipeline:
                if plan.acceptance_criteria is None:
                    console.print(
                        "[dim]Note: listing/listing_report pipeline without acceptance_criteria; "
                        "items are not LLM-validated against goal rules.[/dim]"
                    )
                with console.status("[bold cyan]Extracting listings from pages..."):
                    extracted_listings = await gather_listings_from_traces(
                        llm,
                        all_traces,
                        goal,
                        store=None,
                        max_chars_per_page=research_config.max_chars_per_page_for_extraction,
                        model=effective_model,
                        max_parallel=effective_parallel,
                    )
                n_pages = _unique_crawled_page_count(all_traces)
                console.print(
                    f"[dim]Extracted {len(extracted_listings)} listing(s) from "
                    f"{n_pages} unique page(s)[/dim]"
                )
                listings = filter_stale_listings(extracted_listings)
                stale_n = len(extracted_listings) - len(listings)
                if stale_n:
                    console.print(
                        f"[dim]Filtered {stale_n} stale listing(s); {len(listings)} kept[/dim]"
                    )
                if plan.acceptance_criteria:
                    with console.status("[bold cyan]Validating items against criteria..."):
                        verdicts = await validate_items(
                            llm,
                            listings,
                            plan.acceptance_criteria,
                            goal,
                            model=effective_model,
                        )
                    for v in verdicts:
                        if not v.passed:
                            console.print(
                                f"[dim]FAIL: {v.item.url} — {'; '.join(v.rule_results)}[/dim]"
                            )
                        elif v.unknown_count:
                            console.print(
                                f"[dim]PASS (partial): {v.item.url} — "
                                f"{v.unknown_count} rule(s) unverifiable[/dim]"
                            )
                    fail_n = sum(1 for v in verdicts if not v.passed)
                    if fail_n:
                        console.print(f"[dim]Rejected {fail_n} item(s) by acceptance criteria[/dim]")
                    listings = [v.item for v in verdicts if v.passed]
                final_listings = listings
                if plan.pipeline == PipelineType.listing:
                    report = format_listings_as_list(listings)
                else:
                    if not listings:
                        console.print(
                            "[yellow]No items extracted; synthesizing report from crawl summaries only.[/yellow]"
                        )
                    with console.status("[bold cyan]Summarizing crawls for report..."):
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
                    with console.status("[bold cyan]Generating report from items and summaries..."):
                        report = await synthesize_report_from_listings_and_summaries(
                            llm,
                            persona or "You are a research agent",
                            instructions or goal,
                            listings,
                            batch_summaries,
                            report_focus=plan.report_focus,
                            report_format_instructions=plan.report_format_instructions,
                            model=effective_model,
                        )
            elif research_config.use_batched_summarization:
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
                        report_format_instructions=plan.report_format_instructions,
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
                        report_format_instructions=plan.report_format_instructions,
                        model=effective_model,
                    )

        report_text = report if isinstance(report, str) else str(report)
        if output:
            Path(output).write_text(report_text, encoding="utf-8")
            console.print(f"[green]✓[/green] Report saved to [bold]{output}[/bold]")
            if plan.pipeline == PipelineType.listing_report and final_listings:
                out_path = Path(output)
                listings_path = out_path.with_name(f"{out_path.stem}-listings{out_path.suffix}")
                listings_path.write_text(format_listings_as_list(final_listings), encoding="utf-8")
                console.print(f"[green]✓[/green] Extracted items saved to [bold]{listings_path}[/bold]")
        if show_full or not output:
            if show_full:
                console.print(report_text)
            else:
                preview = report_text[:5000] + ("..." if len(report_text) > 5000 else "")
                console.print(Panel(preview, title="Report", border_style="blue"))
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise

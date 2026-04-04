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
from spiderweb.research.models import PipelineType
from spiderweb.research.agent import (
    aggregate_traces_for_report,
    create_research_plan,
    dedupe_listings,
    format_listings_as_list,
    gather_listings_from_traces,
    generate_more_queries,
    generate_research_queries,
    summarize_traces_in_batches,
    synthesize_report,
    synthesize_report_from_summaries,
    validate_items,
)
from spiderweb.search.content_filter import filter_stale_listings
from spiderweb.search.trace import SearchCrawlTrace

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
        help="Filter by recency (duckduckgo timelimit): d (day), w (week), m (month), y (year).",
    )(f)
    f = click.option("--limit", type=int, default=10, help="Max search results per round. Default: 10.")(f)
    f = click.option("--crawl-provider", type=click.Choice(["http", "crawl4ai", "x"]), default="crawl4ai", help="Crawler. Default: crawl4ai.")(f)
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

    search_config = SearchProviderConfig(
        provider=search_provider,
        limit=limit,
        extra_config={"timelimit": recency} if recency else {},
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
    type=click.IntRange(1, 20),
    default=5,
    help="Max iterative search rounds when using --target.",
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

    search_config = SearchProviderConfig(
        provider=search_provider,
        limit=limit,
        extra_config={"timelimit": recency} if recency else {},
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

        use_listing_pipeline = plan.pipeline == PipelineType.listing

        # Iterative search when target is set and listing pipeline is active
        if target is not None and use_listing_pipeline:
            console.print(f"[dim]Iterative search mode: targeting {target} listings[/dim]")
            if plan.acceptance_criteria is None:
                console.print(
                    "[dim]Note: listing pipeline without acceptance_criteria; "
                    "items are not LLM-validated against goal rules.[/dim]"
                )
            all_listings: list = []
            all_traces: list[SearchCrawlTrace] = []
            queries_used: list[str] = []
            available_queries = list(plan.queries)
            round_num = 0

            while len(all_listings) < target and round_num < research_config.max_search_rounds:
                # Get queries for this round
                if available_queries:
                    round_queries = available_queries[:research_config.queries_per_round]
                    available_queries = available_queries[research_config.queries_per_round:]
                else:
                    console.print(f"[dim]Round {round_num + 1}: Generating more queries...[/dim]")
                    round_queries = await generate_more_queries(
                        llm,
                        goal,
                        queries_already_tried=queries_used,
                        listings_found_so_far=len(all_listings),
                        target_count=target,
                        num_queries=research_config.queries_per_round,
                        model=effective_model,
                    )
                    if not round_queries:
                        console.print("[yellow]No more queries could be generated.[/yellow]")
                        break

                console.print(f"[bold cyan]Round {round_num + 1}:[/bold cyan] Running {len(round_queries)} queries...")

                # Execute queries for this round
                with console.status(f"[bold cyan]Round {round_num + 1}: Executing {len(round_queries)} queries..."):
                    traces = await asyncio.gather(*[run_one(q) for q in round_queries])
                round_traces = [t for t in traces if isinstance(t, SearchCrawlTrace)]
                all_traces.extend(round_traces)
                queries_used.extend(round_queries)

                # Extract and filter listings from this round
                with console.status(f"[bold cyan]Round {round_num + 1}: Extracting listings..."):
                    extracted_listings = await gather_listings_from_traces(
                        llm,
                        round_traces,
                        goal,
                        store=None,
                        max_chars_per_page=research_config.max_chars_per_page_for_extraction,
                        model=effective_model,
                        max_parallel=effective_parallel,
                    )
                n_pages = _unique_crawled_page_count(round_traces)
                console.print(
                    f"[dim]  Extracted {len(extracted_listings)} listing(s) from "
                    f"{n_pages} unique page(s) this round[/dim]"
                )
                filtered_listings = filter_stale_listings(extracted_listings)
                stale_n = len(extracted_listings) - len(filtered_listings)
                if stale_n:
                    console.print(
                        f"[dim]  Filtered {stale_n} stale listing(s); "
                        f"{len(filtered_listings)} kept[/dim]"
                    )

                if plan.acceptance_criteria:
                    with console.status(f"[bold cyan]Round {round_num + 1}: Validating items against criteria..."):
                        verdicts = await validate_items(
                            llm,
                            filtered_listings,
                            plan.acceptance_criteria,
                            goal,
                            model=effective_model,
                        )
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
                    fail_n = sum(1 for v in verdicts if not v.passed)
                    if fail_n:
                        console.print(f"[dim]  Rejected {fail_n} item(s) by acceptance criteria[/dim]")
                    passed_items = [v.item for v in verdicts if v.passed]
                    to_dedupe = passed_items
                else:
                    to_dedupe = filtered_listings

                # Dedupe against already found listings
                deduped = dedupe_listings(to_dedupe, all_listings)
                dup_n = len(to_dedupe) - len(deduped)
                if dup_n:
                    console.print(f"[dim]  Skipped {dup_n} duplicate(s) already in prior rounds[/dim]")
                all_listings.extend(deduped)
                round_num += 1
                console.print(f"[green]✓[/green] Round {round_num}: Found {len(deduped)} new listings ({len(all_listings)} total)")

                if len(all_listings) >= target:
                    console.print(f"[green]✓[/green] Target of {target} listings reached!")
                    break

            if not all_listings:
                console.print("[red]No listings found. Cannot generate report.[/red]")
                return

            report = format_listings_as_list(all_listings)
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

            if use_listing_pipeline:
                if plan.acceptance_criteria is None:
                    console.print(
                        "[dim]Note: listing pipeline without acceptance_criteria; "
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
                report = format_listings_as_list(listings)
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
        if show_full or not output:
            if show_full:
                console.print(report_text)
            else:
                preview = report_text[:5000] + ("..." if len(report_text) > 5000 else "")
                console.print(Panel(preview, title="Report", border_style="blue"))
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise
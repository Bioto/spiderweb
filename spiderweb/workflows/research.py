"""Iterative goal-driven listing research: search, extract, validate, reflect, repeat.

Plain async workflow (same style as ``XSearchExpandWorkflow``): not a text-in/out
``Workflow`` ABC. Callers inject ``run_query`` so orchestration stays testable and avoids
import cycles with ``Spiderweb``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from spiderweb.models.config import ResearchAgentConfig
from spiderweb.research.agent import (
    dedupe_listings,
    format_listings_as_list,
    gather_listings_from_traces,
    reflect_and_plan_next_round,
    validate_items,
)
from spiderweb.research.models import ItemVerdict, ListingItem, ResearchPlan, RoundReflection
from spiderweb.search.content_filter import filter_stale_listings
from spiderweb.search.trace import SearchCrawlTrace

if TYPE_CHECKING:
    from superglue import GlueLLM

ResearchGoalProgressCallback = Callable[[str, int | None, int | None, Any | None], None]


def _unique_crawled_page_count(traces: list[SearchCrawlTrace]) -> int:
    seen: set[str] = set()
    for trace in traces:
        for round_data in trace.rounds:
            for page in round_data.pages:
                seen.add(page.url)
    return len(seen)


@dataclass
class GoalResearchResult:
    """Outcome of :meth:`GoalResearchWorkflow.run_iterative_listings`."""

    goal: str
    plan: ResearchPlan
    listings: list[ListingItem] = field(default_factory=list)
    reflections: list[RoundReflection] = field(default_factory=list)
    report: str = ""
    rounds_completed: int = 0
    traces: list[SearchCrawlTrace] = field(default_factory=list)
    queries_used: list[str] = field(default_factory=list)
    rejected_items: list[ItemVerdict] = field(default_factory=list)


class GoalResearchWorkflow:
    """Multi-round listing search with LLM reflection when plan queries are exhausted."""

    def __init__(self, *, research_config: ResearchAgentConfig | None = None) -> None:
        self._default_rc = research_config or ResearchAgentConfig()

    async def run_iterative_listings(
        self,
        *,
        goal: str,
        plan: ResearchPlan,
        llm_client: GlueLLM,
        run_query: Callable[[str], Awaitable[SearchCrawlTrace]],
        target_listings: int,
        research_config: ResearchAgentConfig | None = None,
        model: str | None = None,
        persona: str | None = None,
        progress_callback: ResearchGoalProgressCallback | None = None,
    ) -> GoalResearchResult:
        """Run search rounds until ``target_listings`` or ``max_search_rounds``.

        Uses ``plan.queries`` first (``queries_per_round`` per round). When exhausted,
        calls :func:`~spiderweb.research.agent.reflect_and_plan_next_round` instead of
        blind query generation.
        """
        rc = research_config or self._default_rc
        effective_model = model or rc.model
        persona_effective = (persona or "a research agent").strip() or "a research agent"

        def notify(step: str, current: int | None = None, total: int | None = None, detail: Any | None = None) -> None:
            if progress_callback:
                progress_callback(step, current, total, detail)

        all_listings: list[ListingItem] = []
        all_traces: list[SearchCrawlTrace] = []
        queries_used: list[str] = []
        rejected_accum: list[ItemVerdict] = []
        reflections: list[RoundReflection] = []
        available_queries = list(plan.queries)
        round_num = 0

        while len(all_listings) < target_listings and round_num < rc.max_search_rounds:
            if available_queries:
                round_queries = available_queries[: rc.queries_per_round]
                available_queries = available_queries[rc.queries_per_round :]
            else:
                notify("reflect_start", round_num + 1, rc.max_search_rounds, None)
                reflection = await reflect_and_plan_next_round(
                    llm_client,
                    goal,
                    plan,
                    all_listings,
                    queries_used,
                    round_num + 1,
                    target_listings,
                    rc.queries_per_round,
                    rejected_verdicts=rejected_accum,
                    persona=persona_effective,
                    model=effective_model,
                )
                reflections.append(reflection)
                notify("reflection", round_num + 1, rc.max_search_rounds, reflection)
                if not reflection.need_more_queries or not reflection.queries:
                    break
                available_queries = list(reflection.queries)
                round_queries = available_queries[: rc.queries_per_round]
                available_queries = available_queries[rc.queries_per_round :]

            notify("round_queries", round_num + 1, len(round_queries), round_queries)

            traces = await asyncio.gather(*[run_query(q) for q in round_queries])
            round_traces = [t for t in traces if isinstance(t, SearchCrawlTrace)]
            all_traces.extend(round_traces)
            queries_used.extend(round_queries)

            extracted_listings = await gather_listings_from_traces(
                llm_client,
                round_traces,
                goal,
                store=None,
                max_chars_per_page=rc.max_chars_per_page_for_extraction,
                model=effective_model,
                max_parallel=rc.max_parallel_crawls,
            )
            extracted_count = len(extracted_listings)
            filtered_listings = filter_stale_listings(extracted_listings)
            stale_dropped = extracted_count - len(filtered_listings)

            if plan.acceptance_criteria:
                verdicts = await validate_items(
                    llm_client,
                    filtered_listings,
                    plan.acceptance_criteria,
                    goal,
                    model=effective_model,
                )
                rejected_accum.extend(v for v in verdicts if not v.passed)
                passed_items = [v.item for v in verdicts if v.passed]
                deduped = dedupe_listings(passed_items, all_listings)
                to_dedupe_count = len(passed_items)
            else:
                verdicts = None
                deduped = dedupe_listings(filtered_listings, all_listings)
                to_dedupe_count = len(filtered_listings)

            dup_vs_prior = to_dedupe_count - len(deduped)
            n_pages = _unique_crawled_page_count(round_traces)

            all_listings.extend(deduped)
            round_num += 1
            notify(
                "round_done",
                round_num,
                target_listings,
                {
                    "extracted_count": extracted_count,
                    "after_stale_count": len(filtered_listings),
                    "stale_dropped": stale_dropped,
                    "deduped_new": len(deduped),
                    "total_listings": len(all_listings),
                    "pages_crawled": n_pages,
                    "dup_skipped": dup_vs_prior,
                    "verdicts": verdicts,
                },
            )

            if len(all_listings) >= target_listings:
                break

        report = format_listings_as_list(all_listings)
        return GoalResearchResult(
            goal=goal,
            plan=plan,
            listings=all_listings,
            reflections=reflections,
            report=report,
            rounds_completed=round_num,
            traces=all_traces,
            queries_used=queries_used,
            rejected_items=rejected_accum,
        )

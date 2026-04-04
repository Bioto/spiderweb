"""Core research agent functions.

Provides query generation, report synthesis, plan creation, and trace aggregation.
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gluellm import GlueLLM

from spiderweb.config import settings
from spiderweb.observability.logging_config import get_logger
from spiderweb.research.models import (
    DEFAULT_REPORT_FORMAT_INSTRUCTIONS,
    AcceptanceCriteria,
    ExpansionDecision,
    ItemValidationBatch,
    ItemVerdict,
    ListingExtractionPayload,
    ListingItem,
    PageListings,
    PipelineType,
    ResearchPlan,
    rule_result_line_is_fail,
)
from spiderweb.research.storage import ResearchContentStore
from spiderweb.search.trace import PageRecord, SearchCrawlTrace
from spiderweb.utils.gluellm_structured import model_from_structured_complete
from spiderweb.utils.path_utils import sanitize_query_for_path

logger = get_logger(__name__)


def _current_datetime_context() -> str:
    """Return a one-line current date/time for inclusion in LLM prompts."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _iter_page_contents(
    traces: list[SearchCrawlTrace],
    store: ResearchContentStore | None,
    max_chars_per_page: int,
):
    """Yield formatted content for each page (deduped by URL) in trace order."""
    seen_urls: set[str] = set()
    for trace in traces:
        for round_data in trace.rounds:
            for page in round_data.pages:
                if page.url in seen_urls:
                    continue
                seen_urls.add(page.url)
                parts = [f"## {page.url}"]
                if page.summary:
                    parts.append(f"\n**Summary:** {page.summary}\n")
                content = None
                if page.content_path and store is not None:
                    content = store.get_content(page.content_path)
                elif page.content_path:
                    try:
                        file_content = Path(page.content_path).read_text(encoding="utf-8")
                        if file_content.startswith("---"):
                            parts_split = file_content.split("---", 2)
                            if len(parts_split) >= 3:
                                file_content = parts_split[2].lstrip("\n")
                        content = file_content
                    except (FileNotFoundError, OSError):
                        logger.warning("Content file not found %s, skipping", page.content_path)
                if not content and page.crawl_result:
                    content = page.crawl_result.markdown or page.crawl_result.content
                if content:
                    if len(content) > max_chars_per_page:
                        content = content[:max_chars_per_page] + "\n\n[... truncated ...]"
                    parts.append(f"\n**Content:**\n{content}\n")
                if page.source_query:
                    parts.append(f"\n*Source query: {page.source_query}*\n")
                yield "\n".join(parts)


def iter_deduped_page_url_and_text(
    traces: list[SearchCrawlTrace],
    store: ResearchContentStore | None,
    max_chars_per_page: int,
):
    """Yield (url, plain_text) for each unique page, for listing extraction."""
    seen_urls: set[str] = set()
    for trace in traces:
        for round_data in trace.rounds:
            for page in round_data.pages:
                if page.url in seen_urls:
                    continue
                seen_urls.add(page.url)
                content = None
                if page.content_path and store is not None:
                    content = store.get_content(page.content_path)
                elif page.content_path:
                    try:
                        file_content = Path(page.content_path).read_text(encoding="utf-8")
                        if file_content.startswith("---"):
                            parts_split = file_content.split("---", 2)
                            if len(parts_split) >= 3:
                                file_content = parts_split[2].lstrip("\n")
                        content = file_content
                    except (FileNotFoundError, OSError):
                        logger.warning("Content file not found %s, skipping", page.content_path)
                if not content and page.crawl_result:
                    content = page.crawl_result.markdown or page.crawl_result.content
                if not content or not str(content).strip():
                    continue
                text = str(content).strip()
                if len(text) > max_chars_per_page:
                    text = text[:max_chars_per_page] + "\n\n[... truncated ...]"
                yield page.url, text


def dedupe_listings(
    new_listings: list[ListingItem],
    existing_listings: list[ListingItem],
) -> list[ListingItem]:
    """Return new_listings that aren't already in existing_listings (by URL)."""
    existing_urls = {listing.url.lower().rstrip("/") for listing in existing_listings}
    deduped = []
    for listing in new_listings:
        normalized_url = listing.url.lower().rstrip("/")
        if normalized_url not in existing_urls:
            deduped.append(listing)
            existing_urls.add(normalized_url)
    return deduped


async def validate_items(
    llm_client: "GlueLLM",
    items: list[ListingItem],
    criteria: AcceptanceCriteria,
    goal: str,
    model: str | None = None,
    *,
    max_batch: int = 20,
) -> list[ItemVerdict]:
    """LLM pass/fail each extracted item against acceptance criteria (batched)."""
    if not items:
        return []
    max_batch = max(1, min(max_batch, 50))
    rules_text = "\n".join(f"{i + 1}. {r}" for i, r in enumerate(criteria.rules))
    all_verdicts: list[ItemVerdict] = []

    for start in range(0, len(items), max_batch):
        batch = items[start : start + max_batch]
        batch_json = json.dumps([li.model_dump() for li in batch], ensure_ascii=False, indent=2)
        prompt = f"""Current date and time: {_current_datetime_context()}

Research goal: {goal}

Acceptance summary: {criteria.description}

Rules — evaluate each rule against the item fields (url, title, price, location, status, details):
{rules_text}

Items to evaluate (array index 0 = first item, 1 = second, ...):
{batch_json}

For each item, return one verdict with item_index equal to its index in the array above.
rule_results must have exactly one short line per rule in the same order as the rules list above. Each line must start with one of:
- PASS: — the item's fields explicitly confirm the rule
- FAIL: — the item's fields explicitly contradict the rule (wrong year, wrong location, automatic when manual required, etc.)
- UNKNOWN: — the relevant data is absent from the item fields; the item neither confirms nor contradicts the rule

Set passed=true when there are ZERO FAIL results (PASS and UNKNOWN are both acceptable).
Set passed=false when there is at least one FAIL.

Do NOT invent facts. If a field is null or empty for that aspect of the rule, use UNKNOWN, not FAIL."""

        try:
            from gluellm.api import structured_complete

            raw = await structured_complete(
                user_message=prompt,
                response_format=ItemValidationBatch,
                system_prompt=(
                    "You validate extracted web items against explicit rules. "
                    "Only FAIL when the item explicitly contradicts a rule. "
                    "Missing or absent data is UNKNOWN, not FAIL. Do not invent facts."
                ),
                model=model,
                timeout=settings.llm_timeout,
            )
            payload = model_from_structured_complete(raw, ItemValidationBatch)
        except Exception as e:
            logger.warning("Structured validate_items failed, falling back: %s", e)
            fallback = (
                prompt
                + '\n\nReturn JSON: {"verdicts": [{"item_index": 0, "passed": true/false, '
                + '"rule_results": ["PASS: ...", "UNKNOWN: ...", "FAIL: ..."]}, ...]} '
                + "— one verdict per item; each rule one line; UNKNOWN when data missing."
            )
            complete_kwargs: dict[str, Any] = {
                "user_message": fallback,
                "temperature": 0.1,
                "timeout": settings.llm_timeout,
            }
            if model is not None:
                complete_kwargs["model"] = model
            try:
                response = await llm_client.complete(**complete_kwargs)
            except TypeError:
                complete_kwargs.pop("temperature", None)
                complete_kwargs.pop("model", None)
                complete_kwargs.pop("timeout", None)
                response = await llm_client.complete(**complete_kwargs)
            content = response.final_response.strip() if hasattr(response, "final_response") else str(response).strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            payload = ItemValidationBatch(**json.loads(content))

        by_idx = {v.item_index: v for v in payload.verdicts if v.item_index >= 0}
        for i, item in enumerate(batch):
            row = by_idx.get(i)
            if row is None:
                all_verdicts.append(
                    ItemVerdict(
                        item=item,
                        passed=False,
                        rule_results=["FAIL: no verdict returned for this item"],
                    )
                )
            else:
                has_fail = any(rule_result_line_is_fail(r) for r in row.rule_results)
                all_verdicts.append(
                    ItemVerdict(item=item, passed=not has_fail, rule_results=list(row.rule_results))
                )

    return all_verdicts


async def extract_listings_from_page(
    llm_client: "GlueLLM",
    page_content: str,
    source_url: str,
    goal: str,
    model: str | None = None,
    *,
    max_body_chars: int = 25000,
) -> PageListings:
    """Extract discrete listings/items from one page's text via structured LLM output."""
    cap = max(1000, min(max_body_chars, 50000))
    body = page_content.strip()
    if len(body) > cap:
        body = body[:cap] + "\n\n[... truncated ...]"

    prompt = f"""Current date and time: {_current_datetime_context()}

Research goal: {goal}

Page URL: {source_url}

This page may be:
- A **single listing** detail page — extract exactly one item (or zero if irrelevant).
- A **search results, category, or market index** page — extract **every** distinct listing visible in the content (10, 20, or more if shown).
- Navigation, error, or empty results — return an empty listings array and a short page_status.

Extract every distinct item that helps satisfy the goal (e.g. vehicles for sale, products).
For each item set:
- url: the most specific URL for that item (from markdown links when present; otherwise the page URL only if a single listing)
- title, price, location, details when visible in the content
- status: one of active, sold, ended, expired, pending, unknown — use **active** only if it is clearly still for sale/available now

If the page has no relevant items (error, empty results, navigation only), return an empty listings array and set page_status briefly (e.g. "no results").

Page content:
---
{body}
---
"""
    try:
        from gluellm.api import structured_complete

        raw = await structured_complete(
            user_message=prompt,
            response_format=ListingExtractionPayload,
            system_prompt=(
                "You extract structured listing rows from web page text. "
                "On index/category pages, output one row per visible listing with real URLs from the page. "
                "Do not invent URLs or prices."
            ),
            model=model,
            timeout=settings.llm_timeout,
        )
        payload = model_from_structured_complete(raw, ListingExtractionPayload)
    except Exception:
        complete_kwargs: dict[str, Any] = {
            "user_message": prompt
            + '\n\nReturn JSON with keys: "listings" (array of objects with url, title, price, location, status, details), '
            + '"page_status" (string or null).',
            "temperature": 0.2,
            "timeout": settings.llm_timeout,
        }
        if model is not None:
            complete_kwargs["model"] = model
        try:
            response = await llm_client.complete(**complete_kwargs)
        except TypeError:
            complete_kwargs.pop("temperature", None)
            complete_kwargs.pop("model", None)
            complete_kwargs.pop("timeout", None)
            response = await llm_client.complete(**complete_kwargs)
        content = response.final_response.strip() if hasattr(response, "final_response") else str(response).strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        payload = ListingExtractionPayload(**json.loads(content))

    return PageListings(
        source_url=source_url,
        listings=list(payload.listings),
        page_status=payload.page_status,
    )


async def gather_listings_from_traces(
    llm_client: "GlueLLM",
    traces: list[SearchCrawlTrace],
    goal: str,
    *,
    store: ResearchContentStore | None,
    max_chars_per_page: int,
    model: str | None,
    max_parallel: int,
) -> list[ListingItem]:
    """Extract listings from all unique pages; dedupe by normalized URL."""
    pages = list(iter_deduped_page_url_and_text(traces, store, max_chars_per_page))
    if not pages:
        return []

    sem = asyncio.Semaphore(max(1, max_parallel))

    async def one(url: str, text: str) -> PageListings | None:
        if len(text.strip()) < 80:
            return None
        async with sem:
            try:
                return await extract_listings_from_page(
                    llm_client,
                    text,
                    url,
                    goal,
                    model=model,
                    max_body_chars=max_chars_per_page,
                )
            except Exception as e:
                logger.warning("Listing extraction failed for %s: %s", url, e)
                return None

    results = await asyncio.gather(*[one(u, t) for u, t in pages])
    seen: set[str] = set()
    merged: list[ListingItem] = []
    for pl in results:
        if pl is None:
            continue
        for li in pl.listings:
            raw_url = (li.url or "").strip()
            if not raw_url:
                continue
            key = raw_url.rstrip("/")
            if key in seen:
                continue
            seen.add(key)
            merged.append(li)
    return merged


def format_listings_as_list(listings: list[ListingItem]) -> str:
    """Format all listings as a simple markdown list without LLM involvement."""
    if not listings:
        return "No listings found."
    lines = []
    for li in listings:
        parts = [li.url]
        if li.price:
            parts.append(li.price)
        if li.location:
            parts.append(li.location)
        if li.details:
            parts.append(li.details)
        lines.append("- " + " — ".join(parts))
    return "\n".join(lines)


async def synthesize_report_from_listings(
    llm_client: "GlueLLM",
    persona: str,
    instructions: str,
    listings: list[ListingItem],
    report_focus: str | None = None,
    report_format_instructions: str | None = None,
    model: str | None = None,
) -> str:
    """Produce the final report from structured listing rows (skips batch summarization)."""
    if not listings:
        return "No matching listings were extracted from the crawled pages."

    format_instruction = report_format_instructions or DEFAULT_REPORT_FORMAT_INSTRUCTIONS
    focus_text = f"\n\nReport Focus: {report_focus}" if report_focus else ""

    rows = []
    for li in listings:
        rows.append(
            f"- url: {li.url}\n  title: {li.title or ''}\n  price: {li.price or ''}\n  "
            f"location: {li.location or ''}\n  status: {li.status or ''}\n  details: {li.details or ''}"
        )
    table = "\n".join(rows)

    prompt = f"""Current date and time: {_current_datetime_context()}

You are {persona}.

Your task: {instructions}{focus_text}

Extracted listings (deduplicated):
{table}

{format_instruction}

Report:"""
    complete_kwargs: dict[str, Any] = {"user_message": prompt, "temperature": 0.2, "timeout": settings.llm_timeout}
    if model is not None:
        complete_kwargs["model"] = model
    try:
        response = await llm_client.complete(**complete_kwargs)
    except TypeError:
        complete_kwargs.pop("temperature", None)
        complete_kwargs.pop("model", None)
        complete_kwargs.pop("timeout", None)
        response = await llm_client.complete(**complete_kwargs)
    if hasattr(response, "final_response"):
        return response.final_response.strip()
    return str(response).strip()


async def generate_research_queries(
    llm_client: "GlueLLM",
    persona: str,
    instructions: str,
    num_queries: int = 5,
    max_queries: int = 10,
    model: str | None = None,
) -> list[str]:
    """Generate research queries from persona and instructions."""
    prompt = f"""Current date and time: {_current_datetime_context()}

You are {persona}.

Your task: {instructions}

Generate as many distinct web search queries as you think are needed for comprehensive coverage, up to a maximum of {max_queries}. The queries should be diverse, use clear searchable language, and avoid redundancy.

Return ONLY a JSON array of query strings, nothing else. Example format:
["query 1", "query 2", "query 3"]

JSON array:"""
    complete_kwargs: dict[str, Any] = {"user_message": prompt, "temperature": 0.7, "timeout": settings.llm_timeout}
    if model is not None:
        complete_kwargs["model"] = model
    try:
        response = await llm_client.complete(**complete_kwargs)
    except TypeError:
        complete_kwargs.pop("temperature", None)
        complete_kwargs.pop("model", None)
        complete_kwargs.pop("timeout", None)
        response = await llm_client.complete(**complete_kwargs)
    if hasattr(response, "final_response"):
        content = response.final_response.strip()
    elif hasattr(response, "content"):
        content = response.content.strip()
    else:
        content = str(response).strip()
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()
    queries = json.loads(content)
    if not isinstance(queries, list):
        raise ValueError(f"Expected list, got {type(queries)}")
    if not all(isinstance(q, str) for q in queries):
        raise ValueError("All queries must be strings")
    if len(queries) > max_queries:
        queries = queries[:max_queries]
    return queries


async def decide_additional_queries_with_reasoning(
    llm_client: "GlueLLM",
    persona: str,
    instructions: str,
    content_preview: str,
    queries_already_run: list[str],
    max_new_queries: int = 10,
    model: str | None = None,
) -> ExpansionDecision:
    """Decide if additional queries are needed with explicit reasoning."""
    queries_list = "\n".join(f"- {q}" for q in queries_already_run)
    prompt = f"""Current date and time: {_current_datetime_context()}

You are {persona}.

Your task: {instructions}

We have already run these web search queries and gathered content:
{queries_list}

Content gathered:
{content_preview}

Review the content. Do we need additional web searches to fully satisfy the research task? If YES, suggest up to {max_new_queries} new queries (different from those already run). If NO, indicate no more queries needed.

Return a JSON object with:
- "reasoning": brief analysis of what we learned and what is still missing
- "need_more_queries": true or false
- "queries": array of new search query strings (empty if need_more_queries is false)

JSON object:"""
    try:
        try:
            from gluellm.api import structured_complete
            raw = await structured_complete(
                user_message=prompt,
                response_format=ExpansionDecision,
                system_prompt=f"You are {persona}.",
                model=model,
                timeout=settings.llm_timeout,
            )
            return model_from_structured_complete(raw, ExpansionDecision)
        except (ImportError, AttributeError, Exception):
            complete_kwargs: dict[str, Any] = {"user_message": prompt, "temperature": 0.5, "timeout": settings.llm_timeout}
            if model is not None:
                complete_kwargs["model"] = model
            try:
                response = await llm_client.complete(**complete_kwargs)
            except TypeError:
                complete_kwargs.pop("temperature", None)
                complete_kwargs.pop("model", None)
                complete_kwargs.pop("timeout", None)
                response = await llm_client.complete(**complete_kwargs)
            if hasattr(response, "final_response"):
                content = response.final_response.strip()
            else:
                content = str(response).strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            data = json.loads(content)
            reasoning = data.get("reasoning", "No reasoning provided.")
            need_more = data.get("need_more_queries", False)
            queries = data.get("queries", [])
            if not isinstance(queries, list):
                queries = []
            queries = [q for q in queries if isinstance(q, str) and q not in queries_already_run][:max_new_queries]
            return ExpansionDecision(reasoning=reasoning, need_more_queries=need_more and len(queries) > 0, queries=queries)
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse LLM response as JSON: %s", e)
        return ExpansionDecision(reasoning=str(e), need_more_queries=False, queries=[])


async def decide_additional_queries(
    llm_client: "GlueLLM",
    persona: str,
    instructions: str,
    content_preview: str,
    queries_already_run: list[str],
    max_new_queries: int = 10,
    model: str | None = None,
) -> list[str]:
    """Convenience wrapper that returns only the queries list."""
    decision = await decide_additional_queries_with_reasoning(
        llm_client=llm_client,
        persona=persona,
        instructions=instructions,
        content_preview=content_preview,
        queries_already_run=queries_already_run,
        max_new_queries=max_new_queries,
        model=model,
    )
    return decision.queries


def spill_trace_to_store(trace: SearchCrawlTrace, store: ResearchContentStore) -> None:
    """Spill trace content into a research content store and clear crawl_result from memory."""
    for round_data in trace.rounds:
        for page in round_data.pages:
            if page.crawl_result is None:
                continue
            content = page.crawl_result.markdown or page.crawl_result.content or ""
            ref = store.save_page_content(page.url, content)
            page.content_path = ref
            page.crawl_result = None


def spill_trace_content_to_dir(trace: SearchCrawlTrace, output_dir: str | Path) -> None:
    """Create a subdir from the trace query and spill content there."""
    from spiderweb.research.storage import MarkdownResearchStore
    slug = sanitize_query_for_path(trace.original_query)
    store = MarkdownResearchStore(Path(output_dir), subdir=slug)
    spill_trace_to_store(trace, store)


async def aggregate_traces_for_report(
    traces: list[SearchCrawlTrace],
    max_chars_per_page: int = 4000,
    store: ResearchContentStore | None = None,
) -> str:
    """Aggregate all trace content into one string for report synthesis. Deduplicates by URL."""
    result_parts = []
    for page_content in _iter_page_contents(traces, store, max_chars_per_page):
        # _iter_page_contents yields one string per page; we don't have URL here, so dedup is inside _iter_page_contents
        result_parts.append(page_content)
    result = "\n\n---\n\n".join(result_parts)
    logger.info("Aggregated %s traces (%s chars total)", len(traces), len(result))
    return result


async def summarize_content_batch(
    llm_client: "GlueLLM",
    persona: str,
    instructions: str,
    batch_content: str,
    batch_label: str,
    model: str | None = None,
) -> str:
    """Summarize one batch of research content into a concise summary."""
    prompt = f"""Current date and time: {_current_datetime_context()}

You are {persona}.

Research task: {instructions}

Below is a batch of research content. Produce a concise structured summary: key facts, findings, source URLs where relevant. Keep it focused for later synthesis.

Research content ({batch_label}):
{batch_content}

Concise summary:"""
    complete_kwargs: dict[str, Any] = {"user_message": prompt, "temperature": 0.2, "timeout": settings.llm_timeout}
    if model is not None:
        complete_kwargs["model"] = model
    try:
        response = await llm_client.complete(**complete_kwargs)
    except TypeError:
        complete_kwargs.pop("temperature", None)
        complete_kwargs.pop("model", None)
        complete_kwargs.pop("timeout", None)
        response = await llm_client.complete(**complete_kwargs)
    if hasattr(response, "final_response"):
        return response.final_response.strip()
    if hasattr(response, "content"):
        return response.content.strip()
    return str(response).strip()


async def summarize_traces_in_batches(
    llm_client: "GlueLLM",
    persona: str,
    instructions: str,
    traces: list[SearchCrawlTrace],
    store: ResearchContentStore | None,
    batch_size_pages: int,
    max_chars_per_page: int = 4000,
    model: str | None = None,
) -> list[str]:
    """Summarize all trace content in page batches; return list of batch summaries."""
    batch: list[str] = []
    summaries: list[str] = []
    batch_num = 0
    for page_content in _iter_page_contents(traces, store, max_chars_per_page):
        batch.append(page_content)
        if len(batch) >= batch_size_pages:
            batch_num += 1
            batch_content = "\n\n---\n\n".join(batch)
            summary = await summarize_content_batch(
                llm_client, persona, instructions, batch_content, f"Batch {batch_num}", model=model
            )
            summaries.append(summary)
            batch = []
    if batch:
        batch_num += 1
        batch_content = "\n\n---\n\n".join(batch)
        summary = await summarize_content_batch(
            llm_client, persona, instructions, batch_content, f"Batch {batch_num}", model=model
        )
        summaries.append(summary)
    return summaries


async def synthesize_report_from_summaries(
    llm_client: "GlueLLM",
    persona: str,
    instructions: str,
    batch_summaries: list[str],
    report_focus: str | None = None,
    report_format_instructions: str | None = None,
    report_schema: type | None = None,
    model: str | None = None,
) -> str | Any:
    """Synthesize a final report from batch summaries."""
    focus_text = f"\n\nReport Focus: {report_focus}" if report_focus else ""
    format_instruction = report_format_instructions or DEFAULT_REPORT_FORMAT_INSTRUCTIONS
    combined = "\n\n---\n\n".join(f"## Summary {i + 1}\n{s}" for i, s in enumerate(batch_summaries))
    prompt = f"""Current date and time: {_current_datetime_context()}

You are {persona}.

Your task: {instructions}{focus_text}

Batch summaries:
{combined}

{format_instruction}

Report:"""
    complete_kwargs: dict[str, Any] = {"user_message": prompt, "temperature": 0.3, "timeout": settings.llm_timeout}
    if model is not None:
        complete_kwargs["model"] = model
    try:
        response = await llm_client.complete(**complete_kwargs)
    except TypeError:
        complete_kwargs.pop("temperature", None)
        complete_kwargs.pop("model", None)
        complete_kwargs.pop("timeout", None)
        response = await llm_client.complete(**complete_kwargs)
    if hasattr(response, "final_response"):
        content = response.final_response.strip()
    else:
        content = str(response).strip()
    if report_schema:
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        return report_schema(**json.loads(content))
    return content


async def synthesize_report(
    llm_client: "GlueLLM",
    persona: str,
    instructions: str,
    aggregated_content: str,
    report_focus: str | None = None,
    report_format_instructions: str | None = None,
    report_schema: type | None = None,
    model: str | None = None,
) -> str | Any:
    """Synthesize a final report from aggregated research content."""
    focus_text = f"\n\nReport Focus: {report_focus}" if report_focus else ""
    format_instruction = report_format_instructions or DEFAULT_REPORT_FORMAT_INSTRUCTIONS
    prompt = f"""Current date and time: {_current_datetime_context()}

You are {persona}.

Your task: {instructions}{focus_text}

Research Content:
{aggregated_content}

{format_instruction}

Report:"""
    complete_kwargs: dict[str, Any] = {"user_message": prompt, "temperature": 0.3, "timeout": settings.llm_timeout}
    if model is not None:
        complete_kwargs["model"] = model
    try:
        response = await llm_client.complete(**complete_kwargs)
    except TypeError:
        complete_kwargs.pop("temperature", None)
        complete_kwargs.pop("model", None)
        complete_kwargs.pop("timeout", None)
        response = await llm_client.complete(**complete_kwargs)
    if hasattr(response, "final_response"):
        content = response.final_response.strip()
    else:
        content = str(response).strip()
    if report_schema:
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        return report_schema(**json.loads(content))
    return content


async def create_research_plan(
    llm_client: "GlueLLM",
    goal: str,
    persona: str | None = None,
    instructions: str | None = None,
    model: str | None = None,
) -> ResearchPlan:
    """Create a research plan from a goal."""
    persona_text = f"You are {persona}." if persona else "You are a research agent."
    instructions_text = f"\n\nAdditional instructions: {instructions}" if instructions else ""
    prompt = f"""Current date and time: {_current_datetime_context()}

{persona_text}

Your goal: {goal}{instructions_text}

You have access to web search and web crawling. Create a research plan with:

1. **queries** — Diverse web search queries that maximize coverage for the goal.
2. **report_focus** — What the final report should emphasize.
3. **report_format_instructions** — Exactly how to format the final output. Match what the user explicitly asked for:
   - "list of listings" / "URLs only" → bullet list of URLs/prices/locations only, no explanations or source commentary
   - "summary" → a few concise paragraphs
   - "research report" / no format specified → structured sections with analysis (well-structured comprehensive markdown)
   Default to a well-structured comprehensive report if the user did not specify a format.
4. **pipeline** — Set **listing** when the user wants discrete extractable items (products, listings, jobs, papers, rentals, etc.) from crawled pages; set **narrative** for research, history, comparisons, or opinion surveys without a per-item list.
5. **acceptance_criteria** — When pipeline is **listing**, REQUIRED: an object with **rules** (non-empty list of short, testable natural-language conditions derived from the goal) and **description** (one sentence summarizing a passing item). Example rules: "Must be located in California", "Salary must be at least $150k", "Publication year must be 2023 or later". For **narrative** pipeline, set acceptance_criteria to null or omit it.
6. **search_strategy** — How to filter irrelevant or stale results:
   - **blocked_domains**: Skip informational or off-topic hosts when they hurt the goal (e.g. wikipedia.org for "find live car listings"; empty for pure research).
   - **preferred_domains**: Optional hosts to crawl first when results include them (e.g. listing or auction sites).
   - **validate_url_liveness**: Usually true — skip clearly dead URLs (404/410) before crawling.
   - **stale_content_patterns**: **Page-level only** — regex strings that mean the whole page is useless (e.g. ``no results``, ``page not found``, ``access denied``, ``captcha``). Do **not** use listing-level phrases like ``sold`` or ``auction ended`` here; those are handled when extracting listings. Matching is case-insensitive; do **not** use invalid flags like ``(?i>word`` — use plain ``word`` or valid ``(?i)word``.
   - **required_content_patterns**: If non-empty, pages must match at least one pattern to count as relevant; empty = no requirement. Same regex rules as stale patterns.
   - **query_exclusions**: DuckDuckGo-style exclusions to append (e.g. "-wikipedia") when they help.
   - **query_site_restrictions**: Optional site: tokens to narrow search.
   - **rationale**: One sentence on why this strategy fits the goal.

Queries should be diverse and maximize coverage."""
    try:
        from gluellm.api import structured_complete
        raw = await structured_complete(
            user_message=prompt,
            response_format=ResearchPlan,
            system_prompt=persona_text,
            model=model,
            timeout=settings.llm_timeout,
        )
        return model_from_structured_complete(raw, ResearchPlan)
    except Exception:
        fallback_prompt = prompt + (
            "\n\nReturn JSON with keys: queries, report_focus, report_format_instructions (optional; "
            "omit to use default comprehensive report format), pipeline (string: \"listing\" or \"narrative\"; "
            "default \"narrative\"), acceptance_criteria (optional object with rules: string[] and description: string; "
            "required when pipeline is listing), rationale (optional), "
            "search_strategy (object with blocked_domains, preferred_domains, validate_url_liveness, "
            "stale_content_patterns, required_content_patterns, query_exclusions, "
            "query_site_restrictions, rationale — use [] or false/\"\" as appropriate). "
            "Legacy key use_listing_extraction (boolean) is accepted if pipeline is omitted: true maps to listing."
        )
        complete_kwargs: dict[str, Any] = {"user_message": fallback_prompt, "temperature": 0.7, "timeout": settings.llm_timeout}
        if model is not None:
            complete_kwargs["model"] = model
        try:
            response = await llm_client.complete(**complete_kwargs)
        except TypeError:
            complete_kwargs.pop("temperature", None)
            complete_kwargs.pop("model", None)
            complete_kwargs.pop("timeout", None)
            response = await llm_client.complete(**complete_kwargs)
        content = response.final_response.strip() if hasattr(response, "final_response") else str(response).strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        return ResearchPlan(**json.loads(content))


async def generate_more_queries(
    llm_client: "GlueLLM",
    goal: str,
    queries_already_tried: list[str],
    listings_found_so_far: int,
    target_count: int,
    num_queries: int = 5,
    model: str | None = None,
) -> list[str]:
    """Generate new search queries to find more listings.

    Called when the iterative search loop hasn't reached the target listing count.
    The LLM is prompted to generate diverse queries that avoid repeating previous ones.
    """
    prompt = f"""Current date and time: {_current_datetime_context()}

You are helping to find more listings for this goal: {goal}

So far we've found {listings_found_so_far} listings but need at least {target_count}.

Previous queries that have already been tried:
{chr(10).join(f"- {q}" for q in queries_already_tried)}

Generate {num_queries} NEW, DIVERSE search queries that:
1. Are different from the queries already tried
2. Target different platforms, phrasing, or filters
3. May find additional listings we haven't discovered yet
4. Avoid repeating the same search engines/sites if they've been exhausted

Return a JSON array of query strings only. Example: ["query 1", "query 2", "query 3"]

JSON array:"""
    complete_kwargs: dict[str, Any] = {
        "user_message": prompt,
        "temperature": 0.8,
        "timeout": settings.llm_timeout,
    }
    if model is not None:
        complete_kwargs["model"] = model
    try:
        response = await llm_client.complete(**complete_kwargs)
    except TypeError:
        complete_kwargs.pop("temperature", None)
        complete_kwargs.pop("model", None)
        complete_kwargs.pop("timeout", None)
        response = await llm_client.complete(**complete_kwargs)
    content = response.final_response.strip() if hasattr(response, "final_response") else str(response).strip()
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()
    try:
        queries = json.loads(content)
        if isinstance(queries, list):
            return [str(q) for q in queries if q]
    except json.JSONDecodeError:
        logger.warning("Failed to parse generate_more_queries response as JSON")
    return []

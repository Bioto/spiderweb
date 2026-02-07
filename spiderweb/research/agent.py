"""Core research agent functions.

Provides query generation, report synthesis, plan creation, and trace aggregation.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gluellm import GlueLLM

from spiderweb.config import settings
from spiderweb.observability.logging_config import get_logger
from spiderweb.research.models import ExpansionDecision, ResearchPlan
from spiderweb.research.storage import ResearchContentStore
from spiderweb.search.trace import PageRecord, SearchCrawlTrace
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
            decision = await structured_complete(
                user_message=prompt,
                response_format=ExpansionDecision,
                system_prompt=f"You are {persona}.",
                model=model,
                timeout=settings.llm_timeout,
            )
            return decision
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
    report_schema: type | None = None,
    model: str | None = None,
) -> str | Any:
    """Synthesize a final report from batch summaries."""
    focus_text = f"\n\nReport Focus: {report_focus}" if report_focus else ""
    combined = "\n\n---\n\n".join(f"## Summary {i + 1}\n{s}" for i, s in enumerate(batch_summaries))
    prompt = f"""Current date and time: {_current_datetime_context()}

You are {persona}.

Your task: {instructions}{focus_text}

Batch summaries:
{combined}

Generate a well-structured markdown report that is comprehensive yet concise, integrates findings, and cites sources where relevant.

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
    report_schema: type | None = None,
    model: str | None = None,
) -> str | Any:
    """Synthesize a final report from aggregated research content."""
    focus_text = f"\n\nReport Focus: {report_focus}" if report_focus else ""
    prompt = f"""Current date and time: {_current_datetime_context()}

You are {persona}.

Your task: {instructions}{focus_text}

Research Content:
{aggregated_content}

Generate a well-structured markdown report that addresses the research task, synthesizes information, and cites sources where relevant.

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

You have access to web search and web crawling. Create a research plan: a list of web search queries and a clear description of what the final report should address. Queries should be diverse and maximize coverage."""
    try:
        from gluellm.api import structured_complete
        plan = await structured_complete(
            user_message=prompt,
            response_format=ResearchPlan,
            system_prompt=persona_text,
            model=model,
            timeout=settings.llm_timeout,
        )
        return plan
    except Exception:
        fallback_prompt = prompt + "\n\nReturn JSON: {\"queries\": [\"q1\", \"q2\", ...], \"report_focus\": \"description\", \"rationale\": \"optional\"}"
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

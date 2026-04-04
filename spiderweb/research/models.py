"""Data models for research agent workflows.

Defines structured types for research plans and results.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from spiderweb.search.trace import SearchCrawlTrace

DEFAULT_REPORT_FORMAT_INSTRUCTIONS = (
    "Generate a well-structured markdown report that is comprehensive yet concise, "
    "integrates findings, and cites sources where relevant."
)


class SearchStrategy(BaseModel):
    """Agent-generated strategy for filtering search results and crawls.

    The research LLM sets these fields based on the user's goal (e.g. block
    Wikipedia for live listings; allow it for historical research).
    """

    blocked_domains: list[str] = Field(
        default_factory=list,
        description=(
            "Host suffixes to skip (e.g. 'wikipedia.org'). "
            "Matches subdomains (e.g. en.wikipedia.org)."
        ),
    )
    preferred_domains: list[str] = Field(
        default_factory=list,
        description=(
            "Host suffixes to prioritize in crawl order (e.g. listing sites). "
            "Does not remove other results; only reorders."
        ),
    )
    validate_url_liveness: bool = Field(
        default=True,
        description="If True, HTTP HEAD check before full crawl; clearly dead URLs (404/410) are skipped.",
    )
    stale_content_patterns: list[str] = Field(
        default_factory=list,
        description=(
            "Page-level regex patterns indicating the ENTIRE page is irrelevant. "
            "Use for signals like 'no results found', 'page not found', 'access denied'. "
            "Do NOT use for listing-level indicators like 'sold' or 'auction ended' — "
            "those are handled by item extraction and per-item filtering when the research pipeline is listing."
        ),
    )
    required_content_patterns: list[str] = Field(
        default_factory=list,
        description=(
            "If non-empty, at least one regex must match page body or the page is excluded from synthesis."
        ),
    )
    query_exclusions: list[str] = Field(
        default_factory=list,
        description="Tokens appended to each search query (e.g. '-sold', '-wikipedia').",
    )
    query_site_restrictions: list[str] = Field(
        default_factory=list,
        description="Site restriction tokens appended to queries (e.g. 'site:example.com').",
    )
    rationale: str = Field(
        default="",
        description="Brief explanation of why this strategy fits the goal.",
    )


class ListingItem(BaseModel):
    """Individual listing (or similar discrete item) extracted from a page."""

    url: str = Field(description="Direct URL to the listing or item")
    title: str | None = Field(default=None, description="Listing title or headline")
    price: str | None = Field(default=None, description="Asking price if shown")
    location: str | None = Field(default=None, description="Location, city, or region")
    status: str | None = Field(
        default=None,
        description="One of: active, sold, ended, expired, pending, unknown — use 'active' for currently for sale",
    )
    details: str | None = Field(
        default=None,
        description="Key details: year, mileage, transmission, etc.",
    )


class PageListings(BaseModel):
    """Listings extracted from a single crawled page."""

    source_url: str = Field(description="URL of the page that was analyzed")
    listings: list[ListingItem] = Field(default_factory=list)
    page_status: str | None = Field(
        default=None,
        description="e.g. 'no results', 'error page', 'navigation only' when there are no extractable listings",
    )


class ListingExtractionPayload(BaseModel):
    """LLM structured output for a single page (source URL added by caller)."""

    listings: list[ListingItem] = Field(default_factory=list)
    page_status: str | None = Field(
        default=None,
        description="Set when the page has no extractable listings (error, empty results, nav only)",
    )


class PipelineType(str, Enum):
    """High-level execution mode chosen by the planning LLM."""

    listing = "listing"
    narrative = "narrative"


class AcceptanceCriteria(BaseModel):
    """LLM-generated acceptance criteria for validating extracted items (any domain)."""

    rules: list[str] = Field(
        description=(
            "Natural-language rules an item MUST satisfy to pass. "
            "Each rule is a single testable condition derived from the goal."
        ),
    )
    description: str = Field(
        description="One-sentence summary of what a passing item looks like",
    )


def rule_result_line_is_fail(line: str) -> bool:
    return line.strip().upper().startswith("FAIL")


def rule_result_line_is_unknown(line: str) -> bool:
    return line.strip().upper().startswith("UNKNOWN")


class ItemVerdict(BaseModel):
    """Pass/fail verdict for one extracted item against AcceptanceCriteria."""

    item: ListingItem
    passed: bool
    rule_results: list[str] = Field(
        default_factory=list,
        description="One line per rule: PASS:, FAIL:, or UNKNOWN: (missing data)",
    )

    @property
    def has_fails(self) -> bool:
        return any(rule_result_line_is_fail(r) for r in self.rule_results)

    @property
    def unknown_count(self) -> int:
        return sum(1 for r in self.rule_results if rule_result_line_is_unknown(r))


class ItemVerdictRow(BaseModel):
    """LLM output row: verdict for one item in a batch (by index)."""

    item_index: int = Field(ge=0, description="0-based index into the batch of items sent in the prompt")
    passed: bool
    rule_results: list[str] = Field(
        default_factory=list,
        description="One line per rule: PASS:, FAIL:, or UNKNOWN: (data absent)",
    )


class ItemValidationBatch(BaseModel):
    """Structured LLM output for a batch of item verdicts."""

    verdicts: list[ItemVerdictRow] = Field(default_factory=list)


class ResearchPlan(BaseModel):
    """Structured plan for research execution.

    Generated by LLM from a goal (and optional persona/instructions).
    Defines what queries to run, how to frame the final report, and how to filter search/crawl.
    """

    queries: list[str] = Field(
        description="List of search/crawl queries to execute for comprehensive coverage",
        min_length=1,
    )
    report_focus: str = Field(
        description="Description of what the final report should address or emphasize",
    )
    report_format_instructions: str = Field(
        default=DEFAULT_REPORT_FORMAT_INSTRUCTIONS,
        description=(
            "How the final output must be formatted. Match the user's ask — e.g. bullet list of URLs only, "
            "short summary paragraphs, or a full structured analysis. The default is a standard report."
        ),
    )
    pipeline: PipelineType = Field(
        default=PipelineType.narrative,
        description=(
            "listing: extract discrete items per page and format as a list; "
            "narrative: batch summarization / synthesis without per-item extraction."
        ),
    )
    acceptance_criteria: AcceptanceCriteria | None = Field(
        default=None,
        description=(
            "When pipeline is listing, rules each extracted item must satisfy (domain-agnostic). "
            "Omit or leave null for narrative pipeline."
        ),
    )
    rationale: str | None = Field(
        default=None,
        description="Optional explanation of why these queries were chosen",
    )
    search_strategy: SearchStrategy = Field(
        default_factory=SearchStrategy,
        description="Filtering and search strategy for this goal (domains, liveness, stale patterns, query tweaks).",
    )

    @model_validator(mode="before")
    @classmethod
    def _legacy_use_listing_extraction(cls, data: Any) -> Any:
        """Map deprecated use_listing_extraction JSON to pipeline when pipeline absent."""
        if not isinstance(data, dict):
            return data
        out = dict(data)
        if "pipeline" not in out and "use_listing_extraction" in out:
            if out.pop("use_listing_extraction", False):
                out["pipeline"] = PipelineType.listing.value
        return out


class ExpansionDecision(BaseModel):
    """LLM output for whether to run more queries and which ones.
    
    Used when evaluating if additional web searches are needed after
    initial crawls, based on the content gathered so far.
    """
    
    reasoning: str = Field(
        description="Brief analysis of the crawled content: what we learned, what is still missing or under-covered relative to the research task."
    )
    need_more_queries: bool = Field(
        description="True if additional web searches would significantly improve the research."
    )
    queries: list[str] = Field(
        default_factory=list,
        description="New search queries to run (empty if need_more_queries is False).",
    )


@dataclass
class ResearchReportResult:
    """Result from research_and_report execution.
    
    Contains the synthesized report, traces from parallel crawls,
    and metadata about the research session.
    """
    
    report: str | Any  # str for markdown, or BaseModel instance for structured
    traces: list[SearchCrawlTrace] = field(default_factory=list)
    queries_used: list[str] = field(default_factory=list)
    aggregated_context_preview: str | None = None


@dataclass
class GoalResult:
    """Result from achieve_goal execution.
    
    Contains both the plan that was generated and the execution results.
    Provides transparency into what the agent decided to do.
    """
    
    plan: ResearchPlan
    report: str | Any  # str for markdown, or BaseModel instance for structured
    traces: list[SearchCrawlTrace] = field(default_factory=list)
    queries_used: list[str] = field(default_factory=list)
    rejected_items: list["ItemVerdict"] = field(default_factory=list)

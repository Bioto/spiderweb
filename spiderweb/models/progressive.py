"""Models for Progressive RAG mode.

Progressive RAG optimizes for faster initial ingestion by creating page summaries
first, then fully processing pages on-demand as they're queried.
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class SummaryStrategy(str, Enum):
    """Strategy for generating page summaries."""

    FIRST_N_CHARS = "first_n_chars"
    LLM_SUMMARY = "llm_summary"
    METADATA_ONLY = "metadata_only"


class ProcessingTrigger(str, Enum):
    """When to trigger full processing of a page."""

    IMMEDIATE = "immediate"  # Process synchronously when queried
    BACKGROUND = "background"  # Queue for async processing
    THRESHOLD = "threshold"  # Process after N query hits


class PageSummary(BaseModel):
    """Summary of a single page in a document.

    Used for initial fast queries before full processing.
    """

    page_id: str = Field(..., description="Unique identifier for this page")
    document_id: str = Field(..., description="Parent document ID")
    page_number: int = Field(..., description="Page number in document")
    summary_text: str = Field(..., description="Summary text content")
    embedding: list[float] | None = Field(None, description="Embedding of summary")
    metadata: dict = Field(default_factory=dict, description="Page metadata")
    is_fully_processed: bool = Field(False, description="Whether page has been fully chunked")
    query_hit_count: int = Field(0, description="Number of times this page was returned in queries")
    last_accessed: datetime = Field(default_factory=datetime.utcnow, description="Last query time")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")


class ProgressiveRAGConfig(BaseModel):
    """Configuration for Progressive RAG mode."""

    summary_strategy: SummaryStrategy = Field(
        default=SummaryStrategy.FIRST_N_CHARS,
        description="How to generate page summaries",
    )
    summary_length: int = Field(
        default=800,
        ge=100,
        le=5000,
        description="Characters to extract for first_n_chars strategy",
    )
    llm_summary_model: str | None = Field(
        None,
        description="LLM model for summary generation (if using llm_summary strategy)",
    )
    processing_trigger: ProcessingTrigger = Field(
        default=ProcessingTrigger.IMMEDIATE,
        description="When to trigger full page processing",
    )
    processing_threshold: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Query hits before processing (if using threshold trigger)",
    )
    summary_collection_suffix: str = Field(
        default="_summaries",
        description="Suffix for summary collection name",
    )
    full_collection_suffix: str = Field(
        default="_full",
        description="Suffix for full chunks collection name",
    )
    enable_summary_caching: bool = Field(
        default=True,
        description="Keep summaries after full processing",
    )


class ProgressiveQueryResult(BaseModel):
    """Result from a progressive RAG query."""

    query: str = Field(..., description="Original query text")
    summary_results: list[PageSummary] = Field(..., description="Matching page summaries")
    full_results: list = Field(default_factory=list, description="Full chunks if available")
    newly_processed_pages: list[int] = Field(
        default_factory=list,
        description="Pages that were fully processed during this query",
    )
    processing_time_ms: float = Field(..., description="Total query time in milliseconds")
    cache_hit: bool = Field(False, description="Whether results came from full cache")



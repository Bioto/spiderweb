"""Result models for processing operations.

These models represent the results of various processing operations
like ingestion, validation, and batch processing.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from spiderweb.models.document import Document


class IngestionResult(BaseModel):
    """Result of ingesting a single document.

    Contains the processed document and metadata about the ingestion.
    """

    document: Document = Field(
        description="The processed document with chunks",
    )
    success: bool = Field(
        default=True,
        description="Whether ingestion was successful",
    )
    chunks_created: int = Field(
        description="Number of chunks created",
    )
    chunks_validated: int = Field(
        description="Number of chunks that passed validation",
    )
    chunks_rejected: int = Field(
        default=0,
        description="Number of chunks rejected by validation",
    )
    chunks_deduplicated: int = Field(
        default=0,
        description="Number of chunks removed as duplicates",
    )
    processing_time_seconds: float = Field(
        description="Total processing time in seconds",
    )
    embedding_time_seconds: float = Field(
        default=0.0,
        description="Time spent generating embeddings",
    )
    errors: list[str] = Field(
        default_factory=list,
        description="List of errors encountered during processing",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="List of warnings during processing",
    )
    graph_entities_written: int | None = Field(
        default=None,
        description="Number of entities written to graph store (None if no graph store)",
    )
    graph_relationships_written: int | None = Field(
        default=None,
        description="Number of relationships written to graph store (None if no graph store)",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "document": {
                    "id": "doc_123",
                    "raw_content": "Sample content",
                    "markdown_content": "# Sample",
                    "chunks": [],
                    "metadata": {},
                },
                "success": True,
                "chunks_created": 10,
                "chunks_validated": 9,
                "chunks_rejected": 1,
                "chunks_deduplicated": 0,
                "processing_time_seconds": 2.5,
                "errors": [],
                "warnings": ["Low quality chunk at index 5"],
            }
        }
    )


class BatchIngestionResult(BaseModel):
    """Result of batch document ingestion.

    Aggregates results from processing multiple documents.
    """

    total_documents: int = Field(
        description="Total number of documents attempted",
    )
    successful_documents: int = Field(
        description="Number of successfully processed documents",
    )
    failed_documents: int = Field(
        description="Number of failed documents",
    )
    total_chunks: int = Field(
        description="Total chunks created across all documents",
    )
    total_chunks_validated: int = Field(
        description="Total chunks validated",
    )
    total_chunks_rejected: int = Field(
        default=0,
        description="Total chunks rejected",
    )
    total_chunks_deduplicated: int = Field(
        default=0,
        description="Total chunks removed as duplicates",
    )
    processing_time_seconds: float = Field(
        description="Total processing time in seconds",
    )
    average_time_per_document: float = Field(
        description="Average processing time per document",
    )
    results: list[IngestionResult] = Field(
        description="Individual results for each document",
    )
    errors: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Errors by document path/identifier",
    )
    started_at: datetime = Field(
        default_factory=datetime.now,
        description="When batch processing started",
    )
    completed_at: datetime | None = Field(
        default=None,
        description="When batch processing completed",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_documents": 100,
                "successful_documents": 98,
                "failed_documents": 2,
                "total_chunks": 1500,
                "total_chunks_validated": 1450,
                "total_chunks_rejected": 50,
                "processing_time_seconds": 120.5,
                "average_time_per_document": 1.2,
                "results": [],
                "errors": {"doc1.pdf": ["Extraction failed"], "doc2.pdf": ["Invalid format"]},
            }
        }
    )


class ValidationResult(BaseModel):
    """Result of validating chunks.

    Contains validation scores and details for a set of chunks.
    """

    chunk_id: str = Field(
        description="ID of the validated chunk",
    )
    passed: bool = Field(
        description="Whether chunk passed validation",
    )
    quality_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Quality score (0-1)",
    )
    coherence_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Coherence score from LLM validation (0-1)",
    )
    information_density: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Information density score (0-1)",
    )
    is_duplicate: bool = Field(
        default=False,
        description="Whether chunk is a duplicate of another",
    )
    duplicate_of: str | None = Field(
        default=None,
        description="ID of the original chunk if this is a duplicate",
    )
    issues: list[str] = Field(
        default_factory=list,
        description="List of validation issues found",
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Recommendations for improving chunk quality",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chunk_id": "chunk_123",
                "passed": True,
                "quality_score": 0.85,
                "coherence_score": 0.90,
                "information_density": 0.75,
                "is_duplicate": False,
                "duplicate_of": None,
                "issues": [],
                "recommendations": [],
            }
        }
    )


class QueryResult(BaseModel):
    """Result of a vector store query.

    Contains the matched chunks and their similarity scores.
    """

    query: str = Field(
        description="The original query string",
    )
    chunks: list[dict[str, Any]] = Field(
        description="Matched chunks with their metadata",
    )
    scores: list[float] = Field(
        description="Similarity scores for each chunk (0-1)",
    )
    execution_time_seconds: float = Field(
        description="Query execution time in seconds",
    )
    total_results: int = Field(
        description="Total number of results returned",
    )
    # Query expansion metadata
    expanded_queries: list[str] | None = Field(
        default=None,
        description="List of expanded queries used (if query expansion was enabled)",
    )
    expansion_strategy: str | None = Field(
        default=None,
        description="Expansion strategy used: 'multi_query' or 'hyde' (if enabled)",
    )
    rrf_scores: list[float] | None = Field(
        default=None,
        description="Reciprocal Rank Fusion scores (if query expansion was enabled)",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "What is machine learning?",
                "chunks": [{"id": "chunk_1", "content": "Machine learning is..."}],
                "scores": [0.92],
                "execution_time_seconds": 0.05,
                "total_results": 1,
            }
        }
    )


class ContextChunk(BaseModel):
    """A chunk returned as context for a match.
    
    Includes position offset relative to match and optional semantic scoring.
    """

    chunk: dict[str, Any] = Field(
        description="The chunk data with content and metadata",
    )
    position_offset: int = Field(
        description="Position relative to match (-2, -1, 0 for match, +1, +2)",
    )
    semantic_score: float | None = Field(
        default=None,
        description="Semantic similarity score if semantic guide was used",
    )
    from_expansion: bool = Field(
        default=False,
        description="True if this chunk was found via adaptive expansion",
    )


class MatchContext(BaseModel):
    """Context for a single match, with expansion metadata.
    
    Contains all context chunks for a match and information about
    how the context was retrieved.
    """

    chunks: list[ContextChunk] = Field(
        description="Context chunks surrounding the match",
    )
    expansion_steps_used: int = Field(
        default=0,
        description="Number of expansion steps needed to find relevant context",
    )
    final_window_size: tuple[int, int] = Field(
        default=(0, 0),
        description="Final window size (before, after) after any expansion",
    )


class QueryResultWithContext(QueryResult):
    """Query result with surrounding context chunks.
    
    Extends QueryResult with context information for each match.
    """

    context_by_match: dict[int, MatchContext] = Field(
        default_factory=dict,
        description="Context chunks grouped by match index",
    )
    all_context_chunks: list[dict[str, Any]] = Field(
        default_factory=list,
        description="All context chunks deduplicated across matches",
    )

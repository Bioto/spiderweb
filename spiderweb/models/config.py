"""Configuration models for various Spiderweb components.

These models define the configuration for chunkers, extractors, validators,
and other pipeline components.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

from spiderweb.models.document import ChunkType


class ChunkerConfig(BaseModel):
    """Configuration for chunking strategies.

    Defines how documents should be split into chunks.
    """

    strategy: ChunkType = Field(
        default=ChunkType.HIERARCHICAL,
        description="Chunking strategy to use",
    )
    max_chunk_size: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Maximum chunk size in characters",
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        le=1000,
        description="Overlap between chunks in characters",
    )
    min_chunk_size: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Minimum chunk size in characters",
    )
    respect_boundaries: bool = Field(
        default=True,
        description="Respect sentence/paragraph boundaries when chunking",
    )
    # Semantic chunking specific
    semantic_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for semantic chunk boundaries",
    )
    # Hierarchical chunking specific
    preserve_structure: bool = Field(
        default=True,
        description="Preserve document structure (headings, sections) in chunks",
    )
    max_hierarchy_depth: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum hierarchy depth for hierarchical chunking",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "strategy": "semantic",
                "max_chunk_size": 1000,
                "chunk_overlap": 200,
                "min_chunk_size": 100,
                "respect_boundaries": True,
                "semantic_threshold": 0.7,
            }
        }


class ExtractorConfig(BaseModel):
    """Configuration for document extraction.

    Defines how documents should be extracted and processed.
    """

    primary_method: Literal["markitdown", "auto"] = Field(
        default="markitdown",
        description="Primary extraction method to use",
    )
    enable_cross_extraction: bool = Field(
        default=False,
        description="Extract with multiple methods for comparison",
    )
    cross_extraction_methods: list[str] = Field(
        default_factory=lambda: ["markitdown"],
        description="Methods to use for cross-extraction",
    )
    preserve_formatting: bool = Field(
        default=True,
        description="Preserve document formatting where possible",
    )
    extract_metadata: bool = Field(
        default=True,
        description="Extract document metadata (author, dates, etc.)",
    )
    extract_images: bool = Field(
        default=False,
        description="Extract and process embedded images",
    )
    ocr_images: bool = Field(
        default=False,
        description="Apply OCR to images (requires OCR dependencies)",
    )
    language: str | None = Field(
        default=None,
        description="Document language (None for auto-detect)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "primary_method": "markitdown",
                "enable_cross_extraction": True,
                "cross_extraction_methods": ["markitdown", "pdfminer"],
                "preserve_formatting": True,
                "extract_metadata": True,
            }
        }


class ValidatorConfig(BaseModel):
    """Configuration for chunk validation.

    Defines validation rules and thresholds for chunk quality.
    """

    enable_validation: bool = Field(
        default=True,
        description="Enable chunk validation",
    )
    min_quality_score: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum quality score for chunks (0-1)",
    )
    enable_deduplication: bool = Field(
        default=True,
        description="Enable semantic deduplication",
    )
    deduplication_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for deduplication (0-1)",
    )
    enable_llm_validation: bool = Field(
        default=False,
        description="Enable LLM-based coherence validation",
    )
    llm_validation_sample_rate: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Fraction of chunks to validate with LLM (0-1)",
    )
    check_completeness: bool = Field(
        default=True,
        description="Check if chunks contain complete sentences/thoughts",
    )
    check_information_density: bool = Field(
        default=True,
        description="Check information density (ratio of meaningful content)",
    )
    min_information_density: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Minimum information density score (0-1)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "enable_validation": True,
                "min_quality_score": 0.5,
                "enable_deduplication": True,
                "deduplication_threshold": 0.95,
                "enable_llm_validation": False,
            }
        }


class BatchConfig(BaseModel):
    """Configuration for batch processing.

    Defines concurrency and error handling for batch operations.
    """

    max_concurrent_extractions: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Maximum concurrent file extractions",
    )
    max_concurrent_embeddings: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum concurrent embedding requests",
    )
    batch_size: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Batch size for vector store operations",
    )
    continue_on_error: bool = Field(
        default=True,
        description="Continue processing if individual documents fail",
    )
    save_checkpoint_interval: int = Field(
        default=100,
        ge=0,
        description="Save progress checkpoint every N documents (0=disabled)",
    )
    checkpoint_path: str | None = Field(
        default=None,
        description="Path to save checkpoints",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "max_concurrent_extractions": 5,
                "max_concurrent_embeddings": 10,
                "batch_size": 100,
                "continue_on_error": True,
            }
        }


class VectorStoreConfig(BaseModel):
    """Configuration for vector store connection.

    Defines connection parameters for various vector stores.
    """

    provider: Literal["memory", "qdrant"] = Field(
        default="memory",
        description="Vector store provider",
    )
    # Qdrant-specific
    host: str = Field(
        default="localhost",
        description="Vector store host",
    )
    port: int = Field(
        default=6333,
        description="Vector store port",
    )
    collection_name: str = Field(
        default="spiderweb_documents",
        description="Collection/index name",
    )
    api_key: str | None = Field(
        default=None,
        description="API key for cloud deployments",
    )
    use_https: bool = Field(
        default=False,
        description="Use HTTPS for connection",
    )
    embedding_dimension: int = Field(
        default=1536,
        description="Dimension of embedding vectors",
    )
    distance_metric: Literal["cosine", "euclidean", "dot"] = Field(
        default="cosine",
        description="Distance metric for similarity search",
    )
    extra_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional provider-specific configuration",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "provider": "qdrant",
                "host": "localhost",
                "port": 6333,
                "collection_name": "my_documents",
                "embedding_dimension": 1536,
                "distance_metric": "cosine",
            }
        }


class ContextWindowConfig(BaseModel):
    """Configuration for retrieving surrounding context for query results.
    
    Enables retrieval of chunks/pages before and after each matched chunk,
    with optional semantic guidance and adaptive expansion.
    """

    enabled: bool = Field(
        default=True,
        description="Whether to include surrounding context chunks",
    )
    chunks_before: int = Field(
        default=2,
        ge=0,
        description="Number of chunks/pages before each match to include",
    )
    chunks_after: int = Field(
        default=2,
        ge=0,
        description="Number of chunks/pages after each match to include",
    )
    context_mode: Literal["page", "chunk"] = Field(
        default="page",
        description="Whether to retrieve by chunk_index or page_number (page-first default)",
    )
    semantic_guide: str | None = Field(
        default=None,
        description="Optional prompt describing what kind of context to look for",
    )
    semantic_boost_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for semantic scoring when guide is provided",
    )
    deduplicate: bool = Field(
        default=True,
        description="Remove duplicate context chunks when multiple matches are close together",
    )
    include_match_in_context: bool = Field(
        default=True,
        description="Include the matched chunk itself in the context results",
    )
    # Adaptive expansion when semantic guide doesn't find good matches
    expand_on_low_score: bool = Field(
        default=True,
        description="Adaptively expand window when semantic scores are below threshold",
    )
    semantic_min_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum semantic similarity score to consider context relevant",
    )
    max_expansion_steps: int = Field(
        default=3,
        ge=0,
        description="Maximum number of expansion attempts when scores are low",
    )
    expansion_step_size: int = Field(
        default=2,
        ge=1,
        description="Number of pages/chunks to add per expansion step",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "enabled": True,
                "chunks_before": 3,
                "chunks_after": 3,
                "context_mode": "page",
                "semantic_guide": "Focus on financial data and metrics",
                "expand_on_low_score": True,
                "semantic_min_score": 0.6,
            }
        }

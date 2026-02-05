"""Configuration models for various Spiderweb components.

These models define the configuration for chunkers, extractors, validators,
and other pipeline components.
"""

from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from spiderweb.config import settings
from spiderweb.models.document import ChunkType


class ChunkerConfig(BaseModel):
    """Configuration for chunking strategies.

    Defines how documents should be split into chunks.

    The strategy field accepts either a ChunkType enum value or a string.
    Strings are used for custom chunkers registered in the chunker_registry.

    Example:
        # Built-in chunker using enum
        config = ChunkerConfig(strategy=ChunkType.HIERARCHICAL)

        # Custom chunker using string
        config = ChunkerConfig(strategy="my-custom-chunker")
    """

    strategy: Union[ChunkType, str] = Field(
        default=ChunkType.HIERARCHICAL,
        description="Chunking strategy to use (ChunkType enum or custom string)",
    )

    @field_validator("strategy", mode="before")
    @classmethod
    def validate_strategy(cls, v: Any) -> Union[ChunkType, str]:
        """Accept both ChunkType enum values and strings."""
        if isinstance(v, ChunkType):
            return v
        if isinstance(v, str):
            # Try to convert to ChunkType if it matches
            try:
                return ChunkType(v)
            except ValueError:
                # Not a known ChunkType, treat as custom strategy name
                return v
        raise ValueError(f"strategy must be ChunkType or str, got {type(v)}")
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

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "strategy": "semantic",
                "max_chunk_size": 1000,
                "chunk_overlap": 200,
                "min_chunk_size": 100,
                "respect_boundaries": True,
                "semantic_threshold": 0.7,
            }
        }
    )


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

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "primary_method": "markitdown",
                "enable_cross_extraction": True,
                "cross_extraction_methods": ["markitdown", "pdfminer"],
                "preserve_formatting": True,
                "extract_metadata": True,
            }
        }
    )


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

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "enable_validation": True,
                "min_quality_score": 0.5,
                "enable_deduplication": True,
                "deduplication_threshold": 0.95,
                "enable_llm_validation": False,
            }
        }
    )


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

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "max_concurrent_extractions": 5,
                "max_concurrent_embeddings": 10,
                "batch_size": 100,
                "continue_on_error": True,
            }
        }
    )


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

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "provider": "qdrant",
                "host": "localhost",
                "port": 6333,
                "collection_name": "my_documents",
                "embedding_dimension": 1536,
                "distance_metric": "cosine",
            }
        }
    )


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

    model_config = ConfigDict(
        json_schema_extra={
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
    )


class QueryExpansionConfig(BaseModel):
    """Configuration for query expansion strategies.
    
    Query expansion improves search recall by generating multiple query
    variations (multi-query) or hypothetical answers (HyDE) and combining
    results using Reciprocal Rank Fusion.
    """

    enabled: bool = Field(
        default=False,
        description="Enable query expansion (opt-in feature)",
    )
    strategy: Literal["multi_query", "hyde"] = Field(
        default="multi_query",
        description="Expansion strategy: multi_query for reformulations, hyde for hypothetical answers",
    )
    custom_prompt: str | None = Field(
        default=None,
        description="Custom prompt for query expansion (overrides default)",
    )
    num_expansions: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of query expansions to generate (for multi-query strategy)",
    )
    include_original: bool = Field(
        default=True,
        description="Include the original query in addition to expanded queries",
    )
    rrf_k: int = Field(
        default=60,
        ge=1,
        description="RRF constant k for rank fusion (higher = less aggressive downranking)",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "enabled": True,
                "strategy": "multi_query",
                "num_expansions": 3,
                "include_original": True,
                "rrf_k": 60,
            }
        }
    )


class CrawlerConfig(BaseModel):
    """Configuration for web crawling.
    
    Defines how URLs should be crawled, including depth control,
    link following, rate limiting, and content extraction.
    
    The provider field accepts any string. Built-in providers are "crawl4ai"
    and "http". Custom crawlers can be registered in the crawler_registry.
    
    Example:
        # Built-in crawler
        config = CrawlerConfig(provider="http")
        
        # Custom crawler
        config = CrawlerConfig(provider="my-custom-crawler")
    """
    
    provider: str = Field(
        default_factory=lambda: settings.default_crawler_provider,
        description="Crawler backend to use (built-in: 'crawl4ai', 'http', or custom)",
    )
    
    # Crawl behavior
    max_depth: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Maximum crawl depth (1 = single page, >1 = follow links)",
    )
    max_pages: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Maximum number of pages to crawl",
    )
    follow_patterns: list[str] = Field(
        default_factory=list,
        description="Regex patterns for links to follow (empty = follow all)",
    )
    exclude_patterns: list[str] = Field(
        default_factory=list,
        description="Regex patterns for links to exclude",
    )
    respect_robots_txt: bool = Field(
        default=True,
        description="Respect robots.txt directives",
    )
    
    # Content handling
    wait_for_js: bool = Field(
        default_factory=lambda: settings.default_crawler_wait_for_js,
        description="Wait for JavaScript rendering (crawl4ai only)",
    )
    timeout_seconds: int = Field(
        default_factory=lambda: settings.default_crawler_timeout,
        ge=1,
        le=300,
        description="Request timeout in seconds",
    )
    extract_markdown: bool = Field(
        default=True,
        description="Convert HTML to markdown",
    )
    
    # Rate limiting
    delay_between_requests: float = Field(
        default_factory=lambda: settings.default_crawler_delay,
        ge=0.0,
        le=10.0,
        description="Delay between requests in seconds",
    )
    max_concurrent: int = Field(
        default_factory=lambda: settings.default_crawler_max_concurrent,
        ge=1,
        le=50,
        description="Maximum concurrent requests",
    )
    
    # Additional settings
    user_agent: str | None = Field(
        default=None,
        description="Custom user agent (None = use default)",
    )
    extra_config: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Additional provider-specific configuration. "
            "For crawl4ai provider, keys are passed through to crawl4ai's CrawlerRunConfig "
            "(e.g. page_timeout, check_robots_txt, js_code, wait_for, css_selector, screenshot, "
            "exclude_external_links, etc.). See crawl4ai documentation for full CrawlerRunConfig options."
        ),
    )
    
    # Crawl relevance filtering
    crawl_relevance_prompt: str | None = Field(
        default=None,
        description=(
            "Optional prompt describing what is good vs bad to crawl. "
            "Used to filter URLs/links before crawling. "
            "Example: 'Good: product pages, pricing, docs. Bad: login, signup, ads, footer links.'"
        ),
    )
    crawl_relevance_use_llm: bool = Field(
        default=True,
        description="Use LLM for relevance filtering (if False, fall back to keyword/heuristic)",
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "provider": "crawl4ai",
                "max_depth": 2,
                "max_pages": 20,
                "wait_for_js": True,
                "extract_markdown": True,
                "delay_between_requests": 1.0,
            }
        }
    )


class CrawlExtractionConfig(BaseModel):
    """Configuration for LLM-powered structured extraction from crawled content.
    
    Enables intelligent extraction using semantic guidance, custom queries,
    and Pydantic schema validation with optional auto-improvement.
    """
    
    enabled: bool = Field(
        default=True,
        description="Enable LLM-powered extraction",
    )
    semantic_guide: str | None = Field(
        default=None,
        description="High-level description of what to extract (e.g., 'Look for product prices and reviews')",
    )
    extraction_query: str | None = Field(
        default=None,
        description="Specific extraction instruction or query",
    )
    output_schema: type[BaseModel] | None = Field(
        default=None,
        description="Pydantic model defining the expected output structure",
    )
    
    # Auto-improve settings
    auto_improve: bool = Field(
        default=False,
        description="Enable iterative improvement of extraction results",
    )
    max_improve_iterations: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of improvement iterations",
    )
    improvement_prompt: str | None = Field(
        default=None,
        description="Custom prompt for guiding improvements (overrides default)",
    )
    
    # Extraction behavior
    include_raw_content: bool = Field(
        default=False,
        description="Include raw HTML/markdown in extraction metadata",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="LLM temperature for extraction (0 = deterministic)",
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "enabled": True,
                "semantic_guide": "Extract product information including price, description, and customer reviews",
                "auto_improve": True,
                "max_improve_iterations": 3,
                "temperature": 0.0,
            }
        }
    )


class SearchProviderConfig(BaseModel):
    """Configuration for web search providers.
    
    Defines which search backend to use and how many results to fetch.
    The provider field accepts any string corresponding to a registered
    search provider in search_provider_registry.
    
    Example:
        # Built-in provider (when implemented)
        config = SearchProviderConfig(provider="duckduckgo", limit=10)
        
        # Custom provider
        config = SearchProviderConfig(provider="my-custom-search")
    """
    
    provider: str = Field(
        default_factory=lambda: settings.default_search_provider,
        description="Search provider backend to use (registered in search_provider_registry)",
    )
    limit: int = Field(
        default_factory=lambda: settings.default_search_limit,
        ge=1,
        le=100,
        description="Maximum number of search results to return",
    )
    extra_config: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Additional provider-specific configuration. "
            "For duckduckgo (ddgs): region, safesearch, timeout, timelimit (d/w/m/y), backend (e.g. duckduckgo, bing, brave), page."
        ),
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "provider": "duckduckgo",
                "limit": 10,
                "extra_config": {
                    "region": "us-en",
                    "safesearch": "moderate",
                    "timelimit": "w",
                    "backend": "duckduckgo",
                },
            }
        }
    )


class SearchDepthConfig(BaseModel):
    """Configuration for multi-round search and "go deeper" strategy.
    
    Controls how many search rounds to run and when to expand queries
    or run additional searches if the answer isn't found.
    """
    
    max_search_rounds: int = Field(
        default_factory=lambda: settings.default_max_search_rounds,
        ge=1,
        le=999999,
        description="Maximum number of search → crawl → decide rounds (use high value for 'run until Ctrl+C')",
    )
    crawl_results_per_round: int = Field(
        default_factory=lambda: settings.default_crawl_per_round,
        ge=1,
        le=200,
        description="How many search-result URLs to crawl per round",
    )
    when_to_go_deeper: Literal["always", "expand_queries", "if_not_found"] = Field(
        default_factory=lambda: settings.default_when_to_go_deeper,
        description=(
            "Strategy for going deeper: "
            "'always' = run all rounds, "
            "'expand_queries' = use query expansion for additional searches, "
            "'if_not_found' = check if answer found, then optionally continue"
        ),
    )
    num_expanded_queries: int = Field(
        default_factory=lambda: settings.default_num_expanded_queries,
        ge=0,
        le=10,
        description="Number of expanded queries to generate per expansion round",
    )
    max_pages_total: int | None = Field(
        default=None,
        ge=1,
        description="Cap total pages crawled across all rounds (None = no limit)",
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "max_search_rounds": 2,
                "crawl_results_per_round": 5,
                "when_to_go_deeper": "expand_queries",
                "num_expanded_queries": 3,
                "max_pages_total": 20,
            }
        }
    )

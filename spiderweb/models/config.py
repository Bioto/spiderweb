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


class ChunkAddOnConfig(BaseModel):
    """Configuration for chunk add-ons.

    Defines which add-ons to run and their options. Add-ons run after
    chunking and can enrich chunks with additional data stored in
    chunk.metadata.extra.

    Example:
        # Enable facts extraction add-on
        config = ChunkAddOnConfig(enabled=["facts"])

        # Enable multiple add-ons with options
        config = ChunkAddOnConfig(
            enabled=["facts", "entities"],
            options={"facts": {"max_facts": 10, "model": "gpt-5.1"}}
        )
    """

    enabled: list[str] = Field(
        default_factory=list,
        description="List of add-on names to enable (e.g., ['facts'])",
    )
    options: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-add-on configuration options (addon_name -> options dict)",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "enabled": ["facts"],
                "options": {"facts": {"max_facts": 10}},
            }
        }
    )


class LangExtractAddOnOptions(BaseModel):
    """Options for the LangExtract chunk add-on (source-grounded entity extraction).

    Pass as ChunkAddOnConfig.options["langextract"]. Requires pip install spiderweb[langextract].
    """

    prompt_description: str = Field(
        default="",
        description="What to extract (e.g. 'Extract people, places, and dates'). Required.",
    )
    examples: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Few-shot examples: list of {text, extractions} where extractions "
        "are list of {extraction_class, extraction_text, attributes}.",
    )
    model_id: str = Field(
        default="gpt-5.1",
        description="LangExtract model id (default gpt-5.1; or gemini-2.5-flash, Ollama gemma2:2b, etc.).",
    )
    api_key: str | None = Field(
        default=None,
        description="Override LANGEXTRACT_API_KEY for this run. Not needed for local Ollama.",
    )
    # Local / Ollama
    model_url: str | None = Field(
        default=None,
        description="Base URL for local LLM (e.g. http://localhost:11434 for Ollama). "
        "When set, LangExtract uses the local provider instead of cloud.",
    )
    fence_output: bool | None = Field(
        default=True,
        description="LangExtract fence_output. True for OpenAI (default); set False for Ollama.",
    )
    use_schema_constraints: bool | None = Field(
        default=False,
        description="LangExtract use_schema_constraints. False for OpenAI/Ollama (default).",
    )
    extraction_passes: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Number of extraction passes for long documents (improves recall).",
    )
    max_workers: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Parallel workers for extraction.",
    )
    max_char_buffer: int | None = Field(
        default=2000,
        description="Max characters per chunk sent to the LLM. Required for long docs to stay under token limits (e.g. 2000 for 8k context). None = send full document (may exceed context).",
    )
    attach_to_chunks: bool = Field(
        default=True,
        description="Attach overlapping extractions to each chunk's metadata.extra.",
    )
    use_markdown_content: bool = Field(
        default=True,
        description="Use document.markdown_content instead of raw_content.",
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
    use_cache: bool = Field(
        default=True,
        description="Use ingest cache to skip unchanged files",
    )
    force: bool = Field(
        default=False,
        description="Force re-processing of all files (bypass cache)",
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

    provider: Literal["memory", "qdrant", "chroma"] = Field(
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


class GraphStoreConfig(BaseModel):
    """Configuration for graph store connection (e.g. Neo4j).

    Used when graph_store_url is set so the pipeline can write
    entities and relationships from add-ons.
    """

    provider: Literal["neo4j"] = Field(
        default="neo4j",
        description="Graph store provider",
    )
    uri: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j connection URI (bolt:// or neo4j://)",
    )
    username: str = Field(
        default="neo4j",
        description="Neo4j username",
    )
    password: str = Field(
        default="",
        description="Neo4j password",
    )
    database: str = Field(
        default="neo4j",
        description="Neo4j database name",
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


class RerankConfig(BaseModel):
    """Configuration for re-ranking search results.

    Re-ranking improves precision by scoring query-document relevance more
    accurately than vector similarity alone.
    """

    enabled: bool = Field(
        default=False,
        description="Enable re-ranking (opt-in feature)",
    )
    model: Literal["cross-encoder", "cohere", "none"] = Field(
        default="cross-encoder",
        description="Re-ranking model: cross-encoder (local), cohere (API), or none",
    )
    model_name: str | None = Field(
        default=None,
        description="Specific model name (e.g., 'cross-encoder/ms-marco-MiniLM-L-6-v2' for cross-encoder)",
    )
    top_k: int | None = Field(
        default=None,
        description="Number of top results to return after reranking (None = return all)",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "enabled": True,
                "model": "cross-encoder",
                "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "top_k": 10,
            }
        }
    )


class HybridConfig(BaseModel):
    """Configuration for hybrid search (BM25 + vector).

    Hybrid search combines keyword-based (BM25) and semantic (vector) retrieval
    for improved recall and precision, especially for queries with specific terms.
    """

    enabled: bool = Field(
        default=False,
        description="Enable hybrid search (opt-in feature)",
    )
    rrf_k: int = Field(
        default=60,
        ge=1,
        description="RRF constant k for rank fusion (higher = less aggressive downranking)",
    )
    bm25_top_k_multiplier: float = Field(
        default=2.0,
        ge=1.0,
        description="Multiplier for BM25 top_k (e.g., 2.0 = retrieve 2x top_k before fusion)",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "enabled": True,
                "rrf_k": 60,
                "bm25_top_k_multiplier": 2.0,
            }
        }
    )


class CrawlerConfig(BaseModel):
    """Configuration for web crawling.
    
    Defines how URLs should be crawled, including depth control,
    link following, rate limiting, and content extraction.
    
    The provider field accepts any string. Built-in providers are "crawl4ai",
    "firecrawl" (hosted Firecrawl scrape API), "http", and "x" (X API v2 for
    x.com/twitter.com status URLs). Custom crawlers
    can be registered in the crawler_registry.
    
    Example:
        # Built-in crawler
        config = CrawlerConfig(provider="http")
        
        # Custom crawler
        config = CrawlerConfig(provider="my-custom-crawler")
    """
    
    provider: str = Field(
        default_factory=lambda: settings.default_crawler_provider,
        description="Crawler backend to use (built-in: 'crawl4ai', 'firecrawl', 'http', 'x', or custom)",
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
    restrict_to_start_domains: bool = Field(
        default=True,
        description=(
            "When True, link-following is restricted to the registered domains "
            "(eTLD+1) of the seed URLs. Prevents the crawler from wandering to "
            "external sites when crawling a specific URL or site. "
            "Set to False for broad multi-site crawls."
        ),
    )
    respect_robots_txt: bool = Field(
        default=True,
        description="Respect robots.txt directives",
    )
    use_sitemap: bool = Field(
        default=False,
        description="Parse sitemap.xml to discover URLs before crawling",
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
    overall_timeout_seconds: int = Field(
        default=60,
        ge=10,
        le=300,
        description="Hard cap on total time per page including scrolling/scanning. Kills runaway scroll loops.",
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
    headers: dict[str, str] = Field(
        default_factory=dict,
        description="Custom HTTP headers to send with requests",
    )
    cookies: dict[str, str] = Field(
        default_factory=dict,
        description="Cookies to send with requests",
    )
    extra_config: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Additional provider-specific configuration. "
            "For crawl4ai provider, keys are passed through to crawl4ai's CrawlerRunConfig "
            "(e.g. page_timeout, check_robots_txt, js_code, wait_for, css_selector, screenshot, "
            "exclude_external_links, etc.). For 'firecrawl' provider, keys are passed to "
            "Firecrawl ScrapeOptions (e.g. only_main_content, proxy); reserved: firecrawl_api_key, "
            "firecrawl_api_url. For 'x' provider, use x_bearer_token and optionally x_scraper_config (XScraperConfig.model_dump())."
        ),
    )

    # --- Browser process tuning (crawl4ai BrowserConfig) ---
    browser_light_mode: bool = Field(
        default=False,
        description=(
            "Disable background browser features (crawl4ai light_mode). "
            "Quickest win for reducing CPU on constrained systems."
        ),
    )
    browser_text_mode: bool = Field(
        default=False,
        description="Disable image and media loading for leaner, faster crawls.",
    )
    browser_config: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Additional crawl4ai BrowserConfig kwargs passed at browser launch "
            "(e.g. extra_args, viewport_width, java_script_enabled, sleep_on_close). "
            "Keys here override browser_light_mode / browser_text_mode if duplicated."
        ),
    )

    # --- Per-page crawl behaviour (crawl4ai CrawlerRunConfig) ---
    wait_until: Literal["domcontentloaded", "load", "networkidle"] = Field(
        default="domcontentloaded",
        description=(
            "Navigation-complete signal. "
            "'domcontentloaded' is fastest; 'networkidle' waits for all network "
            "activity to settle (needed for heavy SPAs but much slower)."
        ),
    )
    scan_full_page: bool = Field(
        default=False,
        description=(
            "Auto-scroll the page to trigger lazy-loaded / infinite-scroll content "
            "before extracting. Slower but necessary for some dynamic pages."
        ),
    )
    scroll_delay: float = Field(
        default=0.2,
        ge=0.0,
        le=10.0,
        description="Seconds to pause between scroll steps when scan_full_page=True.",
    )
    max_scroll_steps: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Maximum scroll iterations when scan_full_page=True. "
            "None = scroll until the full page is loaded."
        ),
    )

    # --- Content targeting and filtering (crawl4ai CrawlerRunConfig) ---
    css_selector: str | None = Field(
        default=None,
        description=(
            "CSS selector to restrict extraction to a specific part of the page "
            "(e.g. 'main', 'article', '#content'). Everything outside is discarded."
        ),
    )
    excluded_tags: list[str] = Field(
        default_factory=list,
        description=(
            "HTML tags to strip before extraction "
            "(e.g. ['nav', 'footer', 'aside', 'script', 'style']). "
            "Reduces noise in the extracted content."
        ),
    )
    word_count_threshold: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Minimum word count for a text block to be included. "
            "Blocks below this threshold are discarded. "
            "None = crawl4ai's built-in default (~200 words)."
        ),
    )
    exclude_external_links: bool = Field(
        default=False,
        description="Strip links pointing outside the current domain from extracted content.",
    )
    remove_overlay_elements: bool = Field(
        default=False,
        description="Attempt to remove modal dialogs and popup overlays before extracting.",
    )
    remove_consent_popups: bool = Field(
        default=False,
        description=(
            "Attempt to dismiss GDPR / cookie-consent banners before extracting "
            "(tries 'Accept All' then falls back to DOM removal)."
        ),
    )
    block_ads: bool = Field(
        default=True,
        description=(
            "Block ad networks, trackers, and analytics requests via Playwright route filtering. "
            "Also blocks fonts and media. Reduces page load time and prevents ad/tracker scripts "
            "from preventing the 'load' event from firing on forum/listing pages."
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


def low_resource_crawler_config(**overrides: Any) -> CrawlerConfig:
    """Return a CrawlerConfig pre-tuned to minimise CPU and memory usage.

    Preset:
    - light_mode + text_mode: disables background features and image loading
    - --disable-gpu / --disable-dev-shm-usage: suppress GPU process and /dev/shm OOM
    - domcontentloaded: fastest page-complete signal
    - scan_full_page=False / excluded_tags for nav noise: lean extraction
    - Small viewport: less rendering work per page

    Any CrawlerConfig keyword argument overrides the preset.

    Example:
        >>> config = low_resource_crawler_config(max_concurrent=1, css_selector="article")
    """
    return CrawlerConfig(
        provider="crawl4ai",
        browser_light_mode=True,
        browser_text_mode=True,
        browser_config={
            "extra_args": ["--disable-gpu", "--disable-dev-shm-usage"],
            "viewport_width": 800,
            "viewport_height": 600,
        },
        wait_until="domcontentloaded",
        scan_full_page=False,
        excluded_tags=["nav", "footer", "aside", "script", "style"],
        remove_overlay_elements=True,
        remove_consent_popups=True,
        **overrides,
    )


class XScraperConfig(BaseModel):
    """Configuration for X (Twitter) scraper: search, user graph, and depth controls.

    Used with the 'x' crawler for search_tweets(), scrape_user(), and related
    operations. Pass via CrawlerConfig(provider='x', extra_config={'x_scraper_config': ...})
    or directly to XCrawler.search() / XCrawler.scrape_user().
    """

    # Search (keywords and hashtags)
    max_search_results: int = Field(
        default=100,
        ge=10,
        le=100,
        description="Max tweets per search request (API cap 100 for recent search)",
    )
    include_parent_tweet: bool = Field(
        default=True,
        description="When a search result is a reply, fetch the parent tweet and include it in results (entire post)",
    )
    search_max_pages: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Max pagination pages per search (each page up to max_search_results)",
    )

    # User graph (followers / following)
    max_followers_per_user: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Max followers to fetch per user per request (API cap 1000)",
    )
    max_following_per_user: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Max following to fetch per user per request (API cap 1000)",
    )
    include_followers: bool = Field(
        default=True,
        description="When scraping a user, include their followers list",
    )
    include_following: bool = Field(
        default=True,
        description="When scraping a user, include their following list",
    )

    # User timeline (tweets)
    include_tweets: bool = Field(
        default=False,
        description="When scraping a user, include their recent tweets (timeline)",
    )
    max_tweets_per_user: int = Field(
        default=10,
        ge=0,
        le=100,
        description="Max tweets to fetch per user when include_tweets is True (0 = off, API cap 100 per request)",
    )

    # Depth control (how many "levels" of accounts to traverse)
    graph_depth: int = Field(
        default=1,
        ge=1,
        le=5,
        description="How many levels deep to traverse (1 = user only + their followers/following; 2+ = recurse into those users)",
    )
    max_users_per_level: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="When graph_depth > 1, max users to expand per level (to avoid explosion)",
    )

    # Rate limiting
    delay_between_requests: float = Field(
        default=0.5,
        ge=0.0,
        le=10.0,
        description="Seconds to wait between X API requests (respect rate limits)",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "max_search_results": 50,
                "graph_depth": 2,
                "max_users_per_level": 20,
                "include_followers": True,
                "include_following": True,
            }
        }
    )


class OpenSkyConfig(BaseModel):
    """Configuration for OpenSky Network flight tracking crawler.

    Used with the 'opensky' crawler. Pass via CrawlerConfig(provider='opensky',
    extra_config={'opensky_config': ...}) or directly to OpenSkyCrawler.search().
    """

    delay_between_requests: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="Seconds to wait between OpenSky API requests (respect rate limits)",
    )
    fetch_flight_history: bool = Field(
        default=True,
        description="When tracking icao24, also fetch flight history (last 2 days)",
    )
    history_lookback_hours: int = Field(
        default=48,
        ge=1,
        le=48,
        description="Hours of flight history to fetch for icao24 queries. OpenSky API max is 48.",
    )
    departures_arrivals_lookback_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Hours to look back for departures/arrivals queries",
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
            "For duckduckgo (ddgs): region, safesearch, timeout, timelimit (d/w/m/y), backend (e.g. duckduckgo, bing, brave), page. "
            "For firecrawl: firecrawl_api_key, firecrawl_api_url (self-hosted), sources (e.g. [\"web\"] or [\"web\",\"news\"]), "
            "location, tbs (Google-style date filter), timeout (request timeout in ms, max 300000), categories, "
            "ignore_invalid_urls, integration, scrape_options."
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


class ResearchAgentConfig(BaseModel):
    """Configuration for research agent workflows.

    Controls query generation, parallel execution, and report synthesis
    for persona-driven and goal-driven research.
    """

    num_queries: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Hint for initial query count (LLM decides actual number up to max_queries)",
    )
    max_queries: int = Field(
        default=50,
        ge=1,
        le=100,
        description="Maximum queries the LLM may suggest per batch (initial or expansion). The LLM decides how many to use up to this limit.",
    )
    max_parallel_crawls: int = Field(
        default=10,
        ge=1,
        le=10,
        description="Maximum concurrent search-crawl executions per batch (caps memory use when many queries are run).",
    )
    max_chars_per_page_for_report: int = Field(
        default=4000,
        ge=100,
        le=50000,
        description="Maximum characters per page when aggregating for report synthesis",
    )
    max_chars_per_page_for_extraction: int = Field(
        default=25000,
        ge=1000,
        le=50000,
        description="Maximum characters per page when extracting listings (higher than report cap for aggregation pages).",
    )
    query_generation_prompt_template: str | None = Field(
        default=None,
        description="Custom prompt template for query generation (overrides default)",
    )
    report_synthesis_prompt_template: str | None = Field(
        default=None,
        description="Custom prompt template for report synthesis (overrides default)",
    )
    max_expansion_rounds: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum number of expansion rounds after initial crawl. Each round runs new queries, then the LLM can suggest further queries from the newly scraped content; 0 = disable expansion.",
    )
    expansion_context_max_chars: int = Field(
        default=12000,
        ge=1000,
        le=50000,
        description="Maximum characters of crawled content to include when deciding on additional queries",
    )
    cache_dir: str | None = Field(
        default=None,
        description="Directory for storing crawled content during research run. If set, content is written to disk and traces are kept lightweight (crawl_result cleared). If None, all content stays in memory.",
    )
    cleanup_cache_after_report: bool = Field(
        default=False,
        description="If True and cache_dir is set, delete the session subdirectory after report synthesis (default False so callers can re-use or inspect cached content).",
    )
    use_batched_summarization: bool = Field(
        default=True,
        description="If True, final report is produced by summarizing content in page batches then synthesizing from summaries (lower memory). If False, use legacy single-pass aggregation + synthesis.",
    )
    summary_batch_size_pages: int = Field(
        default=8,
        ge=1,
        le=50,
        description="Number of pages per batch when use_batched_summarization is True. Each batch is summarized by the LLM; then all batch summaries are synthesized into the final report.",
    )
    model: str | None = Field(
        default=None,
        description="Override LLM model for this research run (e.g. openai:gpt-5.1). When None, uses global default from settings.",
    )

    # Iterative listing search settings
    target_listings: int | None = Field(
        default=None,
        description="Stop searching when this many valid listings are found. None = no target (run all queries once).",
    )
    max_search_rounds: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum search rounds before giving up when target_listings is set.",
    )
    queries_per_round: int = Field(
        default=10,
        ge=1,
        le=20,
        description="Number of queries to execute per round when using iterative search.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "num_queries": 5,
                "max_queries": 50,
                "max_parallel_crawls": 10,
                "max_chars_per_page_for_report": 4000,
                "max_chars_per_page_for_extraction": 25000,
                "max_expansion_rounds": 3,
                "expansion_context_max_chars": 12000,
                "cache_dir": None,
                "cleanup_cache_after_report": False,
                "use_batched_summarization": True,
                "summary_batch_size_pages": 8,
                "model": None,
            }
        }
    )

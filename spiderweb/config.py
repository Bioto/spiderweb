"""Configuration management for Spiderweb.

This module provides configuration management using pydantic-settings,
following the same patterns as gluellm for consistency.
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SpiderwebSettings(BaseSettings):
    """Global settings for Spiderweb.

    Settings can be configured via:
    1. Environment variables (prefixed with SPIDERWEB_)
    2. .env file
    3. Direct instantiation

    Example:
        >>> from spiderweb.config import settings
        >>> print(settings.default_chunker)
        'sliding_window'
    """

    model_config = SettingsConfigDict(
        env_prefix="SPIDERWEB_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Chunking settings
    default_chunker: Literal["hierarchical", "semantic", "sentence", "recursive"] = Field(
        default="hierarchical",
        description="Default chunking strategy to use",
    )
    default_chunk_size: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Default maximum chunk size in characters",
    )
    default_chunk_overlap: int = Field(
        default=200,
        ge=0,
        le=1000,
        description="Default overlap between chunks",
    )

    # Embedding settings (delegated to gluellm)
    embedding_model: str = Field(
        default="openai/text-embedding-3-small",
        description="Default embedding model for semantic operations",
    )
    embedding_dimension: int = Field(
        default=1536,
        description="Embedding vector dimension",
    )
    model: str | None = Field(
        default="openai:gpt-4.1-mini-2025-04-14",
        description="Default LLM model for chat/completion. Used by research and other LLM calls. When None, GlueLLM uses its own default.",
    )
    llm_timeout: float = Field(
        default=300.0,
        ge=1.0,
        le=3600.0,
        description="Timeout in seconds for LLM completion requests (e.g. research summarization). Default 300.",
    )

    # Vector store settings
    default_vector_store: Literal["memory", "qdrant"] = Field(
        default="memory",
        description="Default vector store to use",
    )
    qdrant_host: str = Field(
        default="localhost",
        description="Qdrant server host",
    )
    qdrant_port: int = Field(
        default=6333,
        description="Qdrant server port",
    )
    qdrant_api_key: str | None = Field(
        default=None,
        description="Qdrant API key (for cloud deployments)",
    )
    default_collection_name: str = Field(
        default="spiderweb_documents",
        description="Default Qdrant collection name",
    )

    # Validation settings
    enable_validation: bool = Field(
        default=True,
        description="Enable chunk validation by default",
    )
    min_chunk_quality_score: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum quality score for chunks (0-1)",
    )
    deduplication_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for deduplication (0-1)",
    )
    enable_llm_validation: bool = Field(
        default=False,
        description="Enable LLM-based coherence validation (costs API calls)",
    )

    # Batch processing settings
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
    max_tokens_per_embedding_batch: int = Field(
        default=200000,
        ge=1000,
        le=300000,
        description="Maximum tokens per embedding API call (OpenAI limit is 300k)",
    )
    embedding_batch_size: int = Field(
        default=100,
        ge=1,
        le=2048,
        description="Maximum number of texts per embedding batch",
    )
    batch_size: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Batch size for vector store operations",
    )

    # Extraction settings
    default_extraction_method: Literal["markitdown", "auto"] = Field(
        default="markitdown",
        description="Default extraction method",
    )
    enable_cross_extraction: bool = Field(
        default=False,
        description="Extract with multiple methods for comparison",
    )

    # Logging settings
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    log_file_level: str = Field(
        default="DEBUG",
        description="File logging level",
    )
    log_dir: Path = Field(
        default=Path("logs"),
        description="Directory for log files",
    )
    log_file_name: str = Field(
        default="spiderweb.log",
        description="Log file name",
    )
    log_json_format: bool = Field(
        default=False,
        description="Use JSON format for logs",
    )
    log_max_bytes: int = Field(
        default=10485760,  # 10MB
        description="Maximum log file size in bytes",
    )
    log_backup_count: int = Field(
        default=5,
        description="Number of log backup files to keep",
    )
    log_console_output: bool = Field(
        default=True,
        description="Enable console logging output",
    )

    # Storage settings
    cache_dir: Path = Field(
        default=Path(".spiderweb_cache"),
        description="Directory for caching extracted documents",
    )
    enable_cache: bool = Field(
        default=True,
        description="Enable document extraction caching",
    )

    # Search settings
    default_search_provider: str = Field(
        default="duckduckgo",
        description="Default search provider backend",
    )
    default_search_limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Default maximum number of search results per round",
    )

    # Crawler settings
    default_crawler_provider: str = Field(
        default="crawl4ai",
        description="Default crawler backend",
    )
    default_crawler_delay: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="Default delay between crawl requests in seconds",
    )
    default_crawler_timeout: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Default request timeout in seconds",
    )
    default_crawler_wait_for_js: bool = Field(
        default=True,
        description="Default setting for waiting for JavaScript rendering",
    )
    default_crawler_max_concurrent: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Default maximum concurrent crawl requests",
    )

    # Reddit crawler credentials (OAuth)
    reddit_client_id: str | None = Field(
        default=None,
        description="Reddit API client ID for the 'reddit' crawler. Set SPIDERWEB_REDDIT_CLIENT_ID.",
    )
    reddit_client_secret: str | None = Field(
        default=None,
        description="Reddit API client secret for the 'reddit' crawler. Set SPIDERWEB_REDDIT_CLIENT_SECRET.",
    )
    reddit_user_agent: str = Field(
        default="spiderweb:v0.1 (by /u/spiderweb_bot)",
        description="Reddit API user agent. Set SPIDERWEB_REDDIT_USER_AGENT.",
    )

    # Search depth settings
    default_max_search_rounds: int = Field(
        default=1,
        ge=1,
        description="Default maximum number of search rounds",
    )
    default_crawl_per_round: int = Field(
        default=3,
        ge=1,
        le=200,
        description="Default number of search results to crawl per round",
    )
    default_when_to_go_deeper: Literal["always", "expand_queries", "if_not_found"] = Field(
        default="always",
        description="Default strategy for multi-round search",
    )
    default_num_expanded_queries: int = Field(
        default=2,
        ge=0,
        le=10,
        description="Default number of expanded queries per expansion round",
    )

    # Save defaults
    default_save_format: Literal["markdown", "html", "json", "all"] = Field(
        default="all",
        description="Default format for saved crawl results",
    )
    default_trace_format: Literal["json", "markdown", "jsonl"] = Field(
        default="json",
        description="Default format for search-crawl trace files",
    )

    def get_log_level(self) -> int:
        """Convert log level string to logging constant.

        Returns:
            logging level constant (e.g., logging.INFO)
        """
        return getattr(logging, self.log_level.upper(), logging.INFO)

    def get_file_log_level(self) -> int:
        """Convert file log level string to logging constant.

        Returns:
            logging level constant for file logging
        """
        return getattr(logging, self.log_file_level.upper(), logging.DEBUG)


@lru_cache
def get_settings() -> SpiderwebSettings:
    """Get cached settings instance.

    This function uses lru_cache to ensure only one settings instance exists.

    Returns:
        The global settings instance
    """
    return SpiderwebSettings()


# Global settings instance
settings = get_settings()


def reload_settings() -> SpiderwebSettings:
    """Reload settings from environment/file.

    Useful for testing or when configuration changes at runtime.

    Returns:
        Fresh settings instance
    """
    get_settings.cache_clear()
    return get_settings()

"""Spiderweb - Scalable Document Processing and RAG Pipeline.

Spiderweb is a document processing library built for RAG (Retrieval-Augmented Generation)
applications that provides:

- Multi-strategy document extraction with cross-validation
- Intelligent chunking (semantic, hierarchical, sliding window, etc.)
- Quality validation and semantic deduplication
- Pluggable vector storage (Qdrant, and more)
- Batch processing for hundreds of thousands of files
- Built on superglue for LLM and embedding capabilities
- Extensible registry system for custom chunkers, crawlers, and extractors
- Hook system for intercepting and modifying pipeline data

Quick Start:
    >>> import asyncio
    >>> from spiderweb import ingest, query
    >>> from superglue import GlueLLM
    >>>
    >>> async def main():
    ...     llm = GlueLLM()
    ...
    ...     # Ingest a document
    ...     doc = await ingest("document.pdf", llm_client=llm)
    ...     print(f"Created {len(doc.chunks)} chunks")
    ...
    ...     # Query the vector store
    ...     results = await query("What is the main topic?", llm_client=llm, top_k=5)
    ...     for chunk in results.chunks:
    ...         print(chunk["content"])
    ...
    >>> asyncio.run(main())

Main Components:
    - Spiderweb: Main client class for document processing
    - ingest: Quick document ingestion
    - query: Query the vector store
    - process_directory: Batch process a directory of documents

Configuration:
    - SpiderwebSettings: Global settings manager
    - ChunkerConfig: Chunking strategy configuration
    - ValidatorConfig: Validation pipeline configuration

Extensibility:
    - chunker_registry: Register custom chunkers
    - crawler_registry: Register custom crawlers
    - extractor_registry: Register custom extractors
    - hooks: Register callbacks at pipeline stages (before/after chunk, crawl, extract)
    - HookPoint: Enum of available hook points
    - HookContext: Context passed to hook callbacks
"""

from spiderweb.addons.langextract_addon import LangExtractAddOn
from spiderweb.api import Spiderweb, ingest, process_directory, query
from spiderweb.config import SpiderwebSettings, get_settings, settings
from spiderweb.crawlers.base import Crawler, CrawlResult
from spiderweb.crawlers.crawl4ai import Crawl4AICrawler
from spiderweb.crawlers.extraction import CrawlExtractor
from spiderweb.crawlers.http import HttpCrawler
from spiderweb.crawlers.x import XCrawler, XUserScrapeResult
from spiderweb.hooks import HookContext, HookManager, HookPoint, hooks
from spiderweb.loaders.web_loader import WebLoader
from spiderweb.models.config import (
    ChunkAddOnConfig,
    ChunkerConfig,
    ContextWindowConfig,
    CrawlExtractionConfig,
    CrawlerConfig,
    GraphStoreConfig,
    LangExtractAddOnOptions,
    QueryExpansionConfig,
    ResearchAgentConfig,
    SearchDepthConfig,
    SearchProviderConfig,
    ValidatorConfig,
    VectorStoreConfig,
    XScraperConfig,
    low_resource_crawler_config,
)
from spiderweb.models.graph import Entity, Relationship
from spiderweb.stores.graph_base import GraphStore
from spiderweb.research.models import GoalResult, ResearchPlan, ResearchReportResult, SearchStrategy
from spiderweb.registry import (
    ComponentRegistry,
    chunk_addon_registry,
    chunker_registry,
    crawler_registry,
    extractor_registry,
    search_provider_registry,
)
from spiderweb.search.base import SearchProvider, SearchResult, SearchResultBatch
from spiderweb.workflows.x_search_expand import XSearchExpandResult, XSearchExpandWorkflow

__all__ = [
    # Main API
    "Spiderweb",
    "ingest",
    "query",
    "process_directory",
    # Crawlers
    "Crawler",
    "CrawlResult",
    "HttpCrawler",
    "Crawl4AICrawler",
    "CrawlExtractor",
    "XCrawler",
    "XUserScrapeResult",
    "WebLoader",
    # Configuration
    "SpiderwebSettings",
    "settings",
    "get_settings",
    "ChunkerConfig",
    "ChunkAddOnConfig",
    "LangExtractAddOn",
    "LangExtractAddOnOptions",
    "ValidatorConfig",
    "VectorStoreConfig",
    "GraphStoreConfig",
    "Entity",
    "Relationship",
    "GraphStore",
    "ContextWindowConfig",
    "QueryExpansionConfig",
    "ResearchAgentConfig",
    "CrawlerConfig",
    "low_resource_crawler_config",
    "CrawlExtractionConfig",
    "XScraperConfig",
    "SearchProviderConfig",
    "SearchDepthConfig",
    # Research Agent
    "ResearchPlan",
    "ResearchReportResult",
    "GoalResult",
    "SearchStrategy",
    # Search
    "SearchProvider",
    "SearchResult",
    "SearchResultBatch",
    # Workflows
    "XSearchExpandResult",
    "XSearchExpandWorkflow",
    # Extensibility - Registries
    "ComponentRegistry",
    "chunker_registry",
    "chunk_addon_registry",
    "crawler_registry",
    "extractor_registry",
    "search_provider_registry",
    # Extensibility - Hooks
    "hooks",
    "HookPoint",
    "HookContext",
    "HookManager",
]

from spiderweb._version import get_version

__version__ = get_version()

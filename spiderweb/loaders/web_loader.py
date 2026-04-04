"""Web loader for fetching and processing web content.

Provides a loader for web URLs that mirrors the FileLoader pattern,
integrating crawling and extraction into the document pipeline.
"""

import inspect
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from gluellm import GlueLLM
    from pydantic import BaseModel

from spiderweb.crawlers.base import Crawler, CrawlResult
from spiderweb.crawlers.extraction import CrawlExtractor
from spiderweb.hooks import HookManager, HookPoint, hooks as global_hooks
from spiderweb.models.config import CrawlExtractionConfig, CrawlerConfig
from spiderweb.models.document import Document, DocumentMetadata
from spiderweb.observability.logging_config import get_logger
from spiderweb.registry import crawler_registry

logger = get_logger(__name__)


class WebLoader:
    """Loader for web content - mirrors FileLoader pattern.
    
    Combines web crawling with optional LLM-powered extraction to create
    Document objects that can flow through the standard Spiderweb pipeline.
    
    Example:
        >>> loader = WebLoader(llm_client=llm)
        >>> doc = await loader.load("https://example.com")
        >>> print(f"Loaded {len(doc.raw_content)} characters")
    """
    
    def __init__(
        self,
        crawler: Crawler | None = None,
        extractor: CrawlExtractor | None = None,
        llm_client: "GlueLLM | None" = None,
        crawler_config: CrawlerConfig | None = None,
        extraction_config: CrawlExtractionConfig | None = None,
        hook_manager: HookManager | None = None,
    ):
        """Initialize web loader.
        
        Args:
            crawler: Optional crawler instance (defaults to auto-select based on config)
            extractor: Optional extraction instance for structured extraction
            llm_client: Optional GlueLLM client for extraction
            crawler_config: Optional crawler configuration
            extraction_config: Optional extraction configuration
            hook_manager: Optional hook manager for pipeline hooks (defaults to global hooks)
        """
        self.crawler_config = crawler_config or CrawlerConfig()
        self.extraction_config = extraction_config or CrawlExtractionConfig(enabled=False)
        self.llm_client = llm_client
        self.hooks = hook_manager or global_hooks
        
        # Initialize crawler via registry
        if crawler:
            self.crawler = crawler
        else:
            provider = self.crawler_config.provider
            # Look up crawler class in registry
            if provider in crawler_registry:
                crawler_cls = crawler_registry.get(provider)
                sig = inspect.signature(crawler_cls.__init__)
                if "crawler_config" in sig.parameters:
                    self.crawler = crawler_cls(crawler_config=self.crawler_config)
                else:
                    self.crawler = crawler_cls()
            else:
                # Fallback warning and use crawl4ai
                logger.warning(
                    f"Unknown crawler provider '{provider}', falling back to crawl4ai. "
                    f"Available: {crawler_registry.list()}"
                )
                from spiderweb.crawlers.crawl4ai import Crawl4AICrawler

                self.crawler = Crawl4AICrawler(crawler_config=self.crawler_config)
        
        # Initialize extractor if enabled and LLM client provided
        if self.extraction_config.enabled and llm_client:
            self.extractor = extractor or CrawlExtractor(llm_client, self.extraction_config)
        else:
            self.extractor = None
        
        logger.debug(
            f"Initialized WebLoader with {type(self.crawler).__name__} "
            f"(extraction={'enabled' if self.extractor else 'disabled'})"
        )
    
    def _crawl_result_to_document(
        self,
        result: CrawlResult,
        extracted_data: Union[dict, "BaseModel", None] = None,
    ) -> Document:
        """Convert a CrawlResult to a Document.
        
        Args:
            result: Crawl result from crawler
            extracted_data: Optional structured extraction result
            
        Returns:
            Document with content and metadata
        """
        # Use markdown if available, otherwise raw content
        content = result.markdown if result.markdown else result.content
        
        # Create metadata
        metadata = DocumentMetadata(
            source=result.url,
            file_type="html",
            extraction_method=type(self.crawler).__name__,
            extra={
                "status_code": result.status_code,
                "success": result.success,
                "links_found": len(result.links),
                **result.metadata,
            },
        )
        
        # Add extraction metadata if present
        if extracted_data:
            if hasattr(extracted_data, "model_dump"):
                # Pydantic model
                metadata.extra["extracted_data"] = extracted_data.model_dump()
            else:
                # Dict
                metadata.extra["extracted_data"] = extracted_data
        
        # Create document
        document = Document(
            raw_content=result.content,
            markdown_content=content,
            metadata=metadata,
        )
        
        return document

    async def load(
        self,
        url: str,
        crawler_config: CrawlerConfig | None = None,
        extraction_config: CrawlExtractionConfig | None = None,
        output_schema: type["BaseModel"] | None = None,
    ) -> Document:
        """Load and process content from a URL."""
        config = crawler_config or self.crawler_config
        extract_config = extraction_config or self.extraction_config
        logger.info(f"Loading URL: {url}")
        ctx = await self.hooks.run(HookPoint.BEFORE_CRAWL, url, config=config)
        if ctx.skip:
            raise ValueError(f"Crawl skipped by BEFORE_CRAWL hook for {url}")
        crawl_url = ctx.modified_data if ctx.modified_data is not None else url
        result = await self.crawler.crawl(crawl_url, config)
        ctx = await self.hooks.run(HookPoint.AFTER_CRAWL, result, url=crawl_url, config=config)
        if ctx.modified_data is not None:
            result = ctx.modified_data
        if not result.success:
            raise ValueError(f"Failed to crawl {crawl_url}: {result.error}")
        extracted_data = None
        if extract_config.enabled and self.extractor:
            try:
                logger.debug("Performing LLM extraction on crawled content")
                schema = output_schema or extract_config.output_schema
                extract_content = result.markdown if result.markdown else result.content
                if extract_config.auto_improve and schema:
                    extracted_data, _ = await self.extractor.extract_with_improvement(
                        extract_content,
                        schema=schema,
                        semantic_guide=extract_config.semantic_guide,
                        extraction_query=extract_config.extraction_query,
                    )
                else:
                    extracted_data = await self.extractor.extract(
                        extract_content,
                        schema=schema,
                        semantic_guide=extract_config.semantic_guide,
                        extraction_query=extract_config.extraction_query,
                    )
                logger.info(f"Successfully extracted structured data from {url}")
            except Exception as e:
                logger.warning(f"Extraction failed for {url}: {e}")
        document = self._crawl_result_to_document(result, extracted_data)
        logger.debug(f"Loaded document from {url}: {len(document.raw_content)} characters")
        return document

    async def load_many(
        self,
        urls: list[str],
        crawler_config: CrawlerConfig | None = None,
        extraction_config: CrawlExtractionConfig | None = None,
        output_schema: type["BaseModel"] | None = None,
    ) -> list[Document]:
        """Load and process content from multiple URLs."""
        config = crawler_config or self.crawler_config
        extract_config = extraction_config or self.extraction_config
        logger.info(f"Loading {len(urls)} URLs")
        ctx = await self.hooks.run(HookPoint.BEFORE_CRAWL, urls, config=config, batch=True)
        if ctx.skip:
            logger.warning("Batch crawl skipped by BEFORE_CRAWL hook")
            return []
        crawl_urls = ctx.modified_data if ctx.modified_data is not None else urls
        results = await self.crawler.crawl_many(crawl_urls, config)
        ctx = await self.hooks.run(HookPoint.AFTER_CRAWL, results, urls=crawl_urls, config=config, batch=True)
        if ctx.modified_data is not None:
            results = ctx.modified_data
        documents: list[Document] = []
        for result in results:
            if not result.success:
                logger.warning(f"Skipping failed crawl: {result.url} ({result.error})")
                continue
            extracted_data = None
            if extract_config.enabled and self.extractor:
                try:
                    schema = output_schema or extract_config.output_schema
                    extract_content = result.markdown if result.markdown else result.content
                    extracted_data = await self.extractor.extract(
                        extract_content,
                        schema=schema,
                        semantic_guide=extract_config.semantic_guide,
                        extraction_query=extract_config.extraction_query,
                    )
                except Exception as e:
                    logger.warning(f"Extraction failed for {result.url}: {e}")
            document = self._crawl_result_to_document(result, extracted_data)
            documents.append(document)
        logger.info(f"Successfully loaded {len(documents)} documents from {len(crawl_urls)} starting URLs")
        return documents


def crawl_result_to_document(
    result: CrawlResult,
    extracted_data: Union[dict, "BaseModel", None] = None,
    crawler_name: str = "Crawler",
) -> Document:
    """Convert a CrawlResult to a Document (e.g. for ingestion with entity/topic add-ons).

    Use this to run the full pipeline (chunking, LangExtract, entity-relations, graph)
    on X or other crawl results so entities and topics are parsed and queryable.

    Args:
        result: Crawl result from any crawler (e.g. XCrawler.search or scrape_user).
        extracted_data: Optional structured extraction result to attach.
        crawler_name: Optional crawler class name for metadata (default "Crawler").

    Returns:
        Document with content and metadata (including result.metadata for scoping).
    """
    content = result.markdown if result.markdown else result.content
    metadata = DocumentMetadata(
        source=result.url,
        file_type="html",
        extraction_method=crawler_name,
        extra={
            "status_code": result.status_code,
            "success": result.success,
            "links_found": len(result.links),
            **result.metadata,
        },
    )
    if extracted_data:
        if hasattr(extracted_data, "model_dump"):
            metadata.extra["extracted_data"] = extracted_data.model_dump()
        else:
            metadata.extra["extracted_data"] = extracted_data
    return Document(
        raw_content=result.content,
        markdown_content=content,
        metadata=metadata,
    )


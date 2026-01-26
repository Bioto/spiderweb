"""Web loader for fetching and processing web content.

Provides a loader for web URLs that mirrors the FileLoader pattern,
integrating crawling and extraction into the document pipeline.
"""

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
                self.crawler = crawler_cls()
            else:
                # Fallback warning and use crawl4ai
                logger.warning(
                    f"Unknown crawler provider '{provider}', falling back to crawl4ai. "
                    f"Available: {crawler_registry.list()}"
                )
                from spiderweb.crawlers.crawl4ai import Crawl4AICrawler
                self.crawler = Crawl4AICrawler()
        
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
        """Load and process content from a URL.
        
        Args:
            url: URL to load
            crawler_config: Optional crawler configuration (overrides instance config)
            extraction_config: Optional extraction configuration (overrides instance config)
            output_schema: Optional Pydantic schema for structured extraction
            
        Returns:
            Document with content and metadata
            
        Raises:
            ValueError: If URL is invalid or crawl fails
        """
        config = crawler_config or self.crawler_config
        extract_config = extraction_config or self.extraction_config
        
        logger.info(f"Loading URL: {url}")
        
        # Hook: BEFORE_CRAWL
        ctx = await self.hooks.run(HookPoint.BEFORE_CRAWL, url, config=config)
        if ctx.skip:
            raise ValueError(f"Crawl skipped by BEFORE_CRAWL hook for {url}")
        crawl_url = ctx.modified_data if ctx.modified_data is not None else url
        
        # Crawl the URL
        result = await self.crawler.crawl(crawl_url, config)
        
        # Hook: AFTER_CRAWL
        ctx = await self.hooks.run(HookPoint.AFTER_CRAWL, result, url=crawl_url, config=config)
        if ctx.modified_data is not None:
            result = ctx.modified_data
        
        if not result.success:
            raise ValueError(f"Failed to crawl {crawl_url}: {result.error}")
        
        # Perform extraction if enabled
        extracted_data = None
        if extract_config.enabled and self.extractor:
            try:
                logger.debug("Performing LLM extraction on crawled content")
                
                # Use output_schema if provided, otherwise from config
                schema = output_schema or extract_config.output_schema
                
                # Choose content to extract from (prefer markdown)
                extract_content = result.markdown if result.markdown else result.content
                
                # Perform extraction with auto-improvement if enabled
                if extract_config.auto_improve and schema:
                    extracted_data, improvement_log = await self.extractor.extract_with_improvement(
                        extract_content,
                        schema=schema,
                        semantic_guide=extract_config.semantic_guide,
                        extraction_query=extract_config.extraction_query,
                    )
                    logger.debug(f"Extraction with improvement: {len(improvement_log)} iterations")
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
                # Continue without extraction rather than failing completely
        
        # Convert to Document
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
        """Load and process content from multiple URLs.
        
        Supports link following and concurrent crawling based on crawler config.
        
        Args:
            urls: List of URLs to load
            crawler_config: Optional crawler configuration
            extraction_config: Optional extraction configuration
            output_schema: Optional Pydantic schema for structured extraction
            
        Returns:
            List of Document objects
        """
        config = crawler_config or self.crawler_config
        extract_config = extraction_config or self.extraction_config
        
        logger.info(f"Loading {len(urls)} URLs")
        
        # Hook: BEFORE_CRAWL for batch (pass list of URLs)
        ctx = await self.hooks.run(HookPoint.BEFORE_CRAWL, urls, config=config, batch=True)
        if ctx.skip:
            logger.warning("Batch crawl skipped by BEFORE_CRAWL hook")
            return []
        crawl_urls = ctx.modified_data if ctx.modified_data is not None else urls
        
        # Crawl all URLs (with link following if configured)
        results = await self.crawler.crawl_many(crawl_urls, config)
        
        # Hook: AFTER_CRAWL for batch
        ctx = await self.hooks.run(HookPoint.AFTER_CRAWL, results, urls=crawl_urls, config=config, batch=True)
        if ctx.modified_data is not None:
            results = ctx.modified_data
        
        # Process each result
        documents: list[Document] = []
        
        for result in results:
            if not result.success:
                logger.warning(f"Skipping failed crawl: {result.url} ({result.error})")
                continue
            
            # Perform extraction if enabled
            extracted_data = None
            if extract_config.enabled and self.extractor:
                try:
                    schema = output_schema or extract_config.output_schema
                    extract_content = result.markdown if result.markdown else result.content
                    
                    # Use basic extraction for batch (no auto-improve to save time)
                    extracted_data = await self.extractor.extract(
                        extract_content,
                        schema=schema,
                        semantic_guide=extract_config.semantic_guide,
                        extraction_query=extract_config.extraction_query,
                    )
                except Exception as e:
                    logger.warning(f"Extraction failed for {result.url}: {e}")
            
            # Convert to Document
            document = self._crawl_result_to_document(result, extracted_data)
            documents.append(document)
        
        logger.info(f"Successfully loaded {len(documents)} documents from {len(crawl_urls)} starting URLs")
        
        return documents


"""Spiderweb API - High-level interface for document processing and RAG.

This module provides the main API functions for working with Spiderweb,
following the same patterns as gluellm for a consistent user experience.
"""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gluellm import GlueLLM

    from spiderweb.extractors.base import Extractor

from spiderweb.crawlers.base import CrawlResult
from spiderweb.loaders.web_loader import WebLoader
from spiderweb.models.config import (
    ChunkerConfig,
    ContextWindowConfig,
    CrawlExtractionConfig,
    CrawlerConfig,
    QueryExpansionConfig,
    ValidatorConfig,
    VectorStoreConfig,
)
from spiderweb.models.document import Document
from spiderweb.models.result import BatchIngestionResult, IngestionResult, QueryResult, QueryResultWithContext
from spiderweb.observability.logging_config import get_logger
from spiderweb.pipeline.batch import BatchProcessor
from spiderweb.pipeline.processor import DocumentProcessor
from spiderweb.pipeline.context import ContextRetriever
from spiderweb.pipeline.query_expansion import QueryExpander, reciprocal_rank_fusion

logger = get_logger(__name__)


class Spiderweb:
    """Main Spiderweb client for document processing and RAG.

    Provides a high-level interface for ingesting documents and querying them.

    Example:
        >>> from gluellm import GlueLLM
        >>> from spiderweb import Spiderweb
        >>>
        >>> async with Spiderweb(llm_client=GlueLLM()) as web:
        ...     await web.ingest("document.pdf")
        ...     results = await web.query("What is this document about?")
    """

    def __init__(
        self,
        llm_client: "GlueLLM | None" = None,
        vector_store_url: str | None = None,
        chunker_config: ChunkerConfig | None = None,
        validator_config: ValidatorConfig | None = None,
        store_config: VectorStoreConfig | None = None,
        extractor: "Extractor | None" = None,
    ):
        """Initialize Spiderweb client.

        Args:
            llm_client: GlueLLM client for embeddings and LLM operations
            vector_store_url: Vector store connection URL (e.g., "qdrant://localhost:6333/my_collection")
            chunker_config: Chunking configuration
            validator_config: Validation configuration
            store_config: Vector store configuration
            extractor: Custom document extractor (defaults to MarkitdownExtractor)
        """
        self.llm_client = llm_client
        self.chunker_config = chunker_config
        self.validator_config = validator_config
        self.store_config = store_config
        self.extractor = extractor

        # Parse vector store URL if provided
        if vector_store_url:
            self._parse_store_url(vector_store_url)

        # Initialize components (lazy)
        self._document_processor: DocumentProcessor | None = None
        self._batch_processor: BatchProcessor | None = None

        logger.info("Initialized Spiderweb client")

    def _parse_store_url(self, url: str) -> None:
        """Parse vector store URL and update store config.

        Args:
            url: URL in format "qdrant://host:port/collection"
        """
        # Simple URL parsing
        # Format: provider://host:port/collection
        if "://" not in url:
            raise ValueError(f"Invalid vector store URL format: {url}")

        provider, rest = url.split("://", 1)

        if provider == "qdrant":
            # Parse host:port/collection
            if "/" in rest:
                host_port, collection = rest.split("/", 1)
            else:
                host_port = rest
                collection = "spiderweb_documents"

            if ":" in host_port:
                host, port_str = host_port.split(":", 1)
                port = int(port_str)
            else:
                host = host_port
                port = 6333

            # Update store config
            if not self.store_config:
                self.store_config = VectorStoreConfig()

            self.store_config.provider = "qdrant"
            self.store_config.host = host
            self.store_config.port = port
            self.store_config.collection_name = collection

            logger.debug(f"Parsed Qdrant URL: host={host}, port={port}, collection={collection}")

    @property
    def document_processor(self) -> DocumentProcessor:
        """Get or create document processor."""
        if self._document_processor is None:
            # Lazy initialization
            self._document_processor = DocumentProcessor(
                llm_client=self.llm_client,
                chunker_config=self.chunker_config,
                validator_config=self.validator_config,
                store_config=self.store_config,
                extractor=self.extractor,
            )
        return self._document_processor

    @property
    def batch_processor(self) -> BatchProcessor:
        """Get or create batch processor."""
        if self._batch_processor is None:
            self._batch_processor = BatchProcessor(
                llm_client=self.llm_client,
                document_processor=self.document_processor,
            )
        return self._batch_processor

    async def ingest(self, file_path: str | Path) -> IngestionResult:
        """Ingest a single document.

        Args:
            file_path: Path to the document

        Returns:
            Ingestion result with statistics

        Raises:
            FileNotFoundError: If file doesn't exist
            ExtractionError: If extraction fails
        """
        return await self.document_processor.process(file_path)

    async def ingest_directory(
        self,
        directory: str | Path,
        recursive: bool = True,
        show_progress: bool = True,
    ) -> BatchIngestionResult:
        """Ingest all documents in a directory.

        Args:
            directory: Path to the directory
            recursive: Process subdirectories recursively
            show_progress: Show progress logging

        Returns:
            Batch ingestion result
        """
        return await self.batch_processor.process_directory(
            directory,
            recursive=recursive,
            show_progress=show_progress,
        )

    async def query(
        self,
        query: str,
        top_k: int = 10,
        filter_dict: dict | None = None,
        context_window: ContextWindowConfig | None = None,
        query_expansion: QueryExpansionConfig | None = None,
    ) -> QueryResult | QueryResultWithContext:
        """Query the vector store.

        Args:
            query: Query string
            top_k: Number of results to return
            filter_dict: Optional metadata filters
            context_window: Optional context window configuration for surrounding chunks
            query_expansion: Optional query expansion configuration for improved recall

        Returns:
            Query result with matched chunks, optionally with context

        Raises:
            ValueError: If no LLM client provided
        """
        if not self.llm_client:
            raise ValueError("LLM client required for querying. Provide llm_client when initializing Spiderweb.")

        import time

        start_time = time.time()

        # Handle query expansion if enabled
        expanded_queries = None
        expansion_strategy = None
        rrf_scores_list = None
        
        if query_expansion and query_expansion.enabled:
            logger.info(f"Query expansion enabled with strategy: {query_expansion.strategy}")
            
            # Expand the query
            expander = QueryExpander(self.llm_client, query_expansion)
            expanded_queries = await expander.expand(query)
            expansion_strategy = query_expansion.strategy
            
            logger.debug(f"Expanded into {len(expanded_queries)} queries: {expanded_queries}")
            
            # Search with each expanded query
            all_query_results = []
            for exp_query in expanded_queries:
                # Generate embedding for this query
                embedding_result = await self.llm_client.embed(exp_query)
                query_embedding = embedding_result.embeddings[0]
                
                # Search vector store
                results = await self.document_processor.vector_store.query(
                    embedding=query_embedding,
                    top_k=top_k,
                    filter_dict=filter_dict,
                )
                all_query_results.append(results)
            
            # Combine results using Reciprocal Rank Fusion
            results = reciprocal_rank_fusion(all_query_results, k=query_expansion.rrf_k)
            
            # Limit to top_k after fusion
            results = results[:top_k]
            
        else:
            # Standard single query path
            # Generate query embedding
            embedding_result = await self.llm_client.embed(query)
            query_embedding = embedding_result.embeddings[0]

            # Search vector store
            results = await self.document_processor.vector_store.query(
                embedding=query_embedding,
                top_k=top_k,
                filter_dict=filter_dict,
            )

        execution_time = time.time() - start_time

        # Format results
        chunks = []
        scores = []
        matched_chunks = []

        for chunk, score in results:
            chunks.append(
                {
                    "id": chunk.id,
                    "content": chunk.content,
                    "document_id": chunk.metadata.document_id,
                    "chunk_index": chunk.metadata.chunk_index,
                    "metadata": chunk.metadata.model_dump(),
                }
            )
            scores.append(score)
            matched_chunks.append(chunk)

        # Save RRF scores if expansion was used
        if query_expansion and query_expansion.enabled:
            rrf_scores_list = scores.copy()

        # Retrieve context if requested
        if context_window:
            retriever = ContextRetriever(vector_store=self.document_processor.vector_store)
            
            context_by_match = await retriever.get_context_for_matches(
                matches=matched_chunks,
                config=context_window,
                llm_client=self.llm_client,
            )

            # Collect all unique context chunks
            all_context_chunks = []
            seen_ids = set()
            
            for match_context in context_by_match.values():
                for ctx_chunk in match_context.chunks:
                    chunk_id = ctx_chunk.chunk["id"]
                    if chunk_id not in seen_ids:
                        seen_ids.add(chunk_id)
                        all_context_chunks.append(ctx_chunk.chunk)

            return QueryResultWithContext(
                query=query,
                chunks=chunks,
                scores=scores,
                execution_time_seconds=execution_time,
                total_results=len(chunks),
                context_by_match=context_by_match,
                all_context_chunks=all_context_chunks,
                expanded_queries=expanded_queries,
                expansion_strategy=expansion_strategy,
                rrf_scores=rrf_scores_list,
            )

        return QueryResult(
            query=query,
            chunks=chunks,
            scores=scores,
            execution_time_seconds=execution_time,
            total_results=len(chunks),
            expanded_queries=expanded_queries,
            expansion_strategy=expansion_strategy,
            rrf_scores=rrf_scores_list,
        )

    async def crawl(
        self,
        url: str | list[str],
        crawler_config: CrawlerConfig | None = None,
        extraction_config: CrawlExtractionConfig | None = None,
        output_schema: type | None = None,
        ingest: bool = False,
        save_to: str | Path | None = None,
        save_format: str = "all",
    ) -> CrawlResult | list[CrawlResult] | IngestionResult | BatchIngestionResult:
        """Crawl URL(s) with optional structured extraction and ingestion.
        
        Args:
            url: Single URL or list of URLs to crawl
            crawler_config: Optional crawler configuration
            extraction_config: Optional extraction configuration for LLM-powered data extraction
            output_schema: Optional Pydantic schema for structured extraction
            ingest: If True, ingest crawled content into vector store
            save_to: Optional directory to save crawled content locally
            save_format: Format for saved files - "markdown", "html", "json", or "all"
            
        Returns:
            CrawlResult(s) if ingest=False, otherwise IngestionResult(s)
            
        Example:
            >>> # Basic crawl
            >>> result = await web.crawl("https://example.com")
            >>> print(result.markdown)
            >>>
            >>> # Crawl and save to local files
            >>> result = await web.crawl(
            ...     "https://example.com",
            ...     save_to="./crawled_data",
            ...     save_format="markdown",
            ... )
            >>>
            >>> # Crawl with schema extraction
            >>> from pydantic import BaseModel
            >>> class Product(BaseModel):
            ...     name: str
            ...     price: float
            >>> result = await web.crawl(
            ...     "https://store.com/product",
            ...     output_schema=Product,
            ...     extraction_config=CrawlExtractionConfig(
            ...         semantic_guide="Extract product information",
            ...         auto_improve=True,
            ...     ),
            ... )
            >>>
            >>> # Crawl and ingest
            >>> result = await web.crawl(
            ...     "https://docs.example.com",
            ...     ingest=True,
            ... )
            >>>
            >>> # Crawl, save locally AND ingest
            >>> result = await web.crawl(
            ...     "https://example.com",
            ...     save_to="./backup",
            ...     ingest=True,
            ... )
        """
        # Create web loader
        web_loader = WebLoader(
            llm_client=self.llm_client,
            crawler_config=crawler_config,
            extraction_config=extraction_config,
        )
        
        # Initialize file storage if requested
        storage = None
        if save_to:
            from spiderweb.crawlers.storage import CrawlStorage
            storage = CrawlStorage(output_dir=save_to)
            logger.info(f"Will save crawled content to {save_to}")
        
        is_list = isinstance(url, list)
        urls = url if is_list else [url]
        
        logger.info(f"Crawling {len(urls)} URL(s) (ingest={ingest}, save_to={save_to})")
        
        if not ingest:
            # Just crawl and return raw results
            if is_list or (crawler_config and crawler_config.max_depth > 1):
                # Use crawl_many for multiple URLs or link following
                results = await web_loader.crawler.crawl_many(urls, crawler_config)
                
                # Save to local storage if requested
                if storage:
                    for result in results:
                        if isinstance(result, CrawlResult):
                            storage.save_crawl_result(result, format=save_format)
                    storage.create_index()
                
                return results
            else:
                # Single URL, simple crawl
                result = await web_loader.crawler.crawl(urls[0], crawler_config)
                
                # Save to local storage if requested
                if storage and isinstance(result, CrawlResult):
                    storage.save_crawl_result(result, format=save_format)
                
                return result
        
        else:
            # Crawl and ingest into vector store
            if is_list or (crawler_config and crawler_config.max_depth > 1):
                # Load multiple documents
                documents = await web_loader.load_many(
                    urls,
                    crawler_config=crawler_config,
                    extraction_config=extraction_config,
                    output_schema=output_schema,
                )
                
                # Ingest each document
                results = []
                for doc in documents:
                    # Process through pipeline
                    # Note: We already have the Document, so we skip file loading
                    # and just chunk/embed/store
                    doc.chunks = self.document_processor.chunker.chunk(doc)
                    
                    if self.document_processor.enable_validation and self.document_processor.validator:
                        validation_results = await self.document_processor.validator.validate_batch(doc.chunks)
                        valid_chunks = [
                            chunk for chunk, result in zip(doc.chunks, validation_results)
                            if result.passed
                        ]
                        doc.chunks = valid_chunks
                    
                    if self.document_processor.enable_embedding and self.llm_client:
                        doc.chunks = await self.document_processor._generate_embeddings(doc.chunks)
                    
                    if doc.chunks:
                        chunks_with_embeddings = [c for c in doc.chunks if c.embedding is not None]
                        if chunks_with_embeddings:
                            await self.document_processor.vector_store.upsert(chunks_with_embeddings)
                    
                    results.append(IngestionResult(
                        document=doc,
                        success=True,
                        chunks_created=len(doc.chunks),
                        chunks_validated=len(doc.chunks),
                        chunks_rejected=0,
                        processing_time_seconds=0.0,
                        embedding_time_seconds=0.0,
                        errors=[],
                        warnings=[],
                    ))
                
                # Create batch result
                return BatchIngestionResult(
                    total_documents=len(documents),
                    successful_documents=len(results),
                    failed_documents=0,
                    total_chunks=sum(r.chunks_created for r in results),
                    total_chunks_validated=sum(r.chunks_validated for r in results),
                    total_chunks_rejected=sum(r.chunks_rejected for r in results),
                    processing_time_seconds=sum(r.processing_time_seconds for r in results),
                    average_time_per_document=sum(r.processing_time_seconds for r in results) / len(results) if results else 0.0,
                    errors={},
                )
            
            else:
                # Single document
                doc = await web_loader.load(
                    urls[0],
                    crawler_config=crawler_config,
                    extraction_config=extraction_config,
                    output_schema=output_schema,
                )
                
                # Process through pipeline
                doc.chunks = self.document_processor.chunker.chunk(doc)
                
                if self.document_processor.enable_validation and self.document_processor.validator:
                    validation_results = await self.document_processor.validator.validate_batch(doc.chunks)
                    valid_chunks = [
                        chunk for chunk, result in zip(doc.chunks, validation_results)
                        if result.passed
                    ]
                    doc.chunks = valid_chunks
                
                if self.document_processor.enable_embedding and self.llm_client:
                    doc.chunks = await self.document_processor._generate_embeddings(doc.chunks)
                
                if doc.chunks:
                    chunks_with_embeddings = [c for c in doc.chunks if c.embedding is not None]
                    if chunks_with_embeddings:
                        await self.document_processor.vector_store.upsert(chunks_with_embeddings)
                
                return IngestionResult(
                    document=doc,
                    success=True,
                    chunks_created=len(doc.chunks),
                    chunks_validated=len(doc.chunks),
                    chunks_rejected=0,
                    processing_time_seconds=0.0,
                    embedding_time_seconds=0.0,
                    errors=[],
                    warnings=[],
                )

    async def crawl_and_query(
        self,
        query: str,
        urls: list[str],
        crawler_config: CrawlerConfig | None = None,
        extraction_config: CrawlExtractionConfig | None = None,
        top_k: int = 10,
        filter_dict: dict | None = None,
        context_window: ContextWindowConfig | None = None,
        query_expansion: QueryExpansionConfig | None = None,
    ) -> QueryResult | QueryResultWithContext:
        """Crawl URLs, ingest content temporarily, then query combined with existing store.
        
        This method crawls fresh content from the web, processes it through the pipeline,
        and queries it along with any existing content in the vector store.
        
        Args:
            query: Query string
            urls: List of URLs to crawl for fresh content
            crawler_config: Optional crawler configuration
            extraction_config: Optional extraction configuration
            top_k: Number of results to return
            filter_dict: Optional metadata filters
            context_window: Optional context window configuration
            query_expansion: Optional query expansion configuration
            
        Returns:
            Query result with matched chunks from both fresh and existing content
            
        Example:
            >>> results = await web.crawl_and_query(
            ...     query="What are the latest pricing changes?",
            ...     urls=["https://company.com/pricing"],
            ...     extraction_config=CrawlExtractionConfig(
            ...         semantic_guide="Focus on pricing tiers and recent updates",
            ...     ),
            ... )
            >>> for chunk in results.chunks:
            ...     print(chunk["content"])
        """
        if not self.llm_client:
            raise ValueError("LLM client required for crawl_and_query")
        
        logger.info(f"Crawl and query: crawling {len(urls)} URLs then querying with: {query}")
        
        # Crawl and ingest the URLs
        await self.crawl(
            url=urls,
            crawler_config=crawler_config,
            extraction_config=extraction_config,
            ingest=True,
        )
        
        # Now query the vector store (includes fresh content)
        return await self.query(
            query=query,
            top_k=top_k,
            filter_dict=filter_dict,
            context_window=context_window,
            query_expansion=query_expansion,
        )

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        # Cleanup if needed
        pass


# Convenience functions for quick usage (gluellm-style)


async def ingest(
    file_path: str | Path,
    llm_client: "GlueLLM | None" = None,
    chunker_config: ChunkerConfig | None = None,
) -> Document:
    """Quick document ingestion.

    Args:
        file_path: Path to the document
        llm_client: Optional GlueLLM client for embeddings
        chunker_config: Optional chunker configuration

    Returns:
        Processed document with chunks

    Example:
        >>> from gluellm import GlueLLM
        >>> from spiderweb import ingest
        >>>
        >>> doc = await ingest("document.pdf", llm_client=GlueLLM())
        >>> print(f"Created {len(doc.chunks)} chunks")
    """
    processor = DocumentProcessor(
        llm_client=llm_client,
        chunker_config=chunker_config,
    )

    result = await processor.process(file_path, store_chunks=False)
    return result.document


async def process_directory(
    directory: str | Path,
    llm_client: "GlueLLM | None" = None,
    chunker: str = "sliding_window",
    store: str = "memory",
    recursive: bool = True,
) -> BatchIngestionResult:
    """Quick batch processing of a directory.

    Args:
        directory: Path to the directory
        llm_client: Optional GlueLLM client for embeddings
        chunker: Chunking strategy ('sliding_window', 'semantic', 'hierarchical')
        store: Vector store type ('memory', 'qdrant')
        recursive: Process subdirectories recursively

    Returns:
        Batch ingestion result

    Example:
        >>> from gluellm import GlueLLM
        >>> from spiderweb import process_directory
        >>>
        >>> result = await process_directory(
        ...     "/path/to/docs",
        ...     llm_client=GlueLLM(),
        ...     chunker="semantic",
        ... )
        >>> print(f"Processed {result.successful_documents} documents")
    """
    # Create chunker config
    from spiderweb.models.document import ChunkType

    chunker_config = ChunkerConfig(strategy=ChunkType(chunker))

    # Create store config
    store_config = VectorStoreConfig(provider=store)

    # Create client
    web = Spiderweb(
        llm_client=llm_client,
        chunker_config=chunker_config,
        store_config=store_config,
    )

    return await web.ingest_directory(directory, recursive=recursive)


async def query(
    query_text: str,
    llm_client: "GlueLLM | None" = None,
    vector_store_url: str | None = None,
    top_k: int = 10,
    context_window: ContextWindowConfig | None = None,
    query_expansion: QueryExpansionConfig | None = None,
) -> QueryResult | QueryResultWithContext:
    """Quick vector store query.

    Args:
        query_text: Query string
        llm_client: GlueLLM client for embeddings
        vector_store_url: Vector store connection URL
        top_k: Number of results
        context_window: Optional context window configuration
        query_expansion: Optional query expansion configuration

    Returns:
        Query result, optionally with context

    Example:
        >>> from gluellm import GlueLLM
        >>> from spiderweb import query
        >>> from spiderweb.models.config import QueryExpansionConfig
        >>>
        >>> results = await query(
        ...     "What is machine learning?",
        ...     llm_client=GlueLLM(),
        ...     top_k=5,
        ...     query_expansion=QueryExpansionConfig(enabled=True),
        ... )
        >>> for chunk in results.chunks:
        ...     print(chunk["content"])
    """
    web = Spiderweb(
        llm_client=llm_client,
        vector_store_url=vector_store_url,
    )

    return await web.query(
        query_text,
        top_k=top_k,
        context_window=context_window,
        query_expansion=query_expansion,
    )

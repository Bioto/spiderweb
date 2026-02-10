"""Spiderweb API - High-level interface for document processing and RAG.

This module provides the main API functions for working with Spiderweb,
following the same patterns as gluellm for a consistent user experience.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Literal, overload

if TYPE_CHECKING:
    from gluellm import GlueLLM

    from spiderweb.extractors.base import Extractor

from spiderweb.config import settings
from spiderweb.crawlers.base import CrawlResult
from spiderweb.hooks import hooks as global_hooks
from spiderweb.loaders.web_loader import WebLoader
from spiderweb.models.config import (
    ChunkAddOnConfig,
    ChunkerConfig,
    ContextWindowConfig,
    CrawlExtractionConfig,
    CrawlerConfig,
    GraphStoreConfig,
    HybridConfig,
    QueryExpansionConfig,
    RerankConfig,
    SearchDepthConfig,
    SearchProviderConfig,
    ValidatorConfig,
    VectorStoreConfig,
)
from spiderweb.models.document import Document
from spiderweb.models.result import BatchIngestionResult, IngestionResult, QueryResult, QueryResultWithContext
from spiderweb.observability.logging_config import get_logger
from spiderweb.pipeline.batch import BatchProcessor
from spiderweb.pipeline.context import ContextRetriever
from spiderweb.pipeline.processor import DocumentProcessor
from spiderweb.pipeline.query_expansion import QueryExpander, reciprocal_rank_fusion
from spiderweb.registry import chunker_registry, crawler_registry, extractor_registry, search_provider_registry
from spiderweb.search.base import SearchResult, SearchResultBatch
from spiderweb.search.relevance import filter_by_relevance
from spiderweb.search.trace import FilteredCandidate, PageRecord, SearchCrawlTrace, SearchRound
from spiderweb.search.writers import write_trace
from spiderweb.utils.path_utils import sanitize_query_for_path

logger = get_logger(__name__)


def parse_graph_store_url(url: str) -> GraphStoreConfig:
    """Parse graph store URL and return GraphStoreConfig.

    Args:
        url: URL in format "neo4j://user:password@host:7687" or "neo4j://host:7687"

    Returns:
        GraphStoreConfig for the parsed URL.
    """
    if "://" not in url:
        raise ValueError(f"Invalid graph store URL format: {url}")

    provider, rest = url.split("://", 1)
    if provider not in ("neo4j", "bolt"):
        raise ValueError(f"Unsupported graph store provider: {provider}")

    username = "neo4j"
    password = ""
    if "@" in rest:
        auth, host_port = rest.split("@", 1)
        if ":" in auth:
            username, password = auth.split(":", 1)
        else:
            username = auth
    else:
        host_port = rest

    if ":" in host_port:
        host, port_str = host_port.rsplit(":", 1)
        port = int(port_str)
    else:
        host = host_port
        port = 7687

    uri = f"bolt://{host}:{port}"
    return GraphStoreConfig(
        provider="neo4j",
        uri=uri,
        username=username,
        password=password,
    )


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

    Extensibility:
        Register custom components via class-level registries::

            # Register a custom chunker
            Spiderweb.chunkers.register("my-chunker", MyChunkerClass)

            # Register a custom crawler
            Spiderweb.crawlers.register("my-crawler", MyCrawlerClass)

            # Add a pipeline hook
            Spiderweb.hooks.register(HookPoint.AFTER_CHUNK, my_callback)
    """

    # Class-level access to component registries for extensibility
    chunkers = chunker_registry
    crawlers = crawler_registry
    extractors = extractor_registry
    hooks = global_hooks

    def __init__(
        self,
        llm_client: "GlueLLM | None" = None,
        vector_store_url: str | None = None,
        graph_store_url: str | None = None,
        chunker_config: ChunkerConfig | None = None,
        validator_config: ValidatorConfig | None = None,
        store_config: VectorStoreConfig | None = None,
        graph_store_config: GraphStoreConfig | None = None,
        extractor: "Extractor | None" = None,
        chunk_add_ons: list[str] | None = None,
        chunk_addon_config: ChunkAddOnConfig | None = None,
    ):
        """Initialize Spiderweb client.

        Args:
            llm_client: GlueLLM client for embeddings and LLM operations
            vector_store_url: Vector store connection URL (e.g., "qdrant://localhost:6333/my_collection")
            graph_store_url: Graph store URL (e.g., "neo4j://user:pass@localhost:7687")
            chunker_config: Chunking configuration
            validator_config: Validation configuration
            store_config: Vector store configuration
            graph_store_config: Graph store configuration (overridden by graph_store_url if set)
            extractor: Custom document extractor (defaults to MarkitdownExtractor)
            chunk_add_ons: List of add-on names to enable (e.g., ["facts", "langextract"])
            chunk_addon_config: Chunk add-on configuration
        """
        self.llm_client = llm_client
        self.chunker_config = chunker_config
        self.validator_config = validator_config
        self.store_config = store_config
        self.graph_store_config = graph_store_config
        self.extractor = extractor
        self.chunk_add_ons = chunk_add_ons
        self.chunk_addon_config = chunk_addon_config

        # Parse vector store URL if provided
        if vector_store_url:
            self._parse_store_url(vector_store_url)

        # Parse graph store URL if provided
        if graph_store_url:
            self._parse_graph_store_url(graph_store_url)

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
        elif provider == "chroma":
            # Parse persist_directory/collection or just collection
            if "/" in rest:
                persist_dir, collection = rest.split("/", 1)
            else:
                persist_dir = None
                collection = rest or "spiderweb_documents"

            # Update store config
            if not self.store_config:
                self.store_config = VectorStoreConfig()

            self.store_config.provider = "chroma"
            self.store_config.collection_name = collection
            # Store persist_directory in extra_config for now (could add to VectorStoreConfig)
            if persist_dir:
                if not hasattr(self.store_config, "extra_config"):
                    self.store_config.extra_config = {}
                self.store_config.extra_config["persist_directory"] = persist_dir

            logger.debug(f"Parsed Chroma URL: persist_directory={persist_dir}, collection={collection}")

    def _parse_graph_store_url(self, url: str) -> None:
        """Parse graph store URL and set graph store config."""
        self.graph_store_config = parse_graph_store_url(url)
        logger.debug(
            f"Parsed graph store URL: uri={self.graph_store_config.uri}, "
            f"user={self.graph_store_config.username}"
        )

    @property
    def document_processor(self) -> DocumentProcessor:
        """Get or create document processor."""
        if self._document_processor is None:
            graph_store = None
            if self.graph_store_config:
                try:
                    from spiderweb.stores.neo4j import Neo4jGraphStore

                    graph_store = Neo4jGraphStore.from_config(self.graph_store_config)
                    logger.debug("Initialized Neo4j graph store for document processor")
                except ImportError as e:
                    raise ImportError(
                        "Graph store is configured but the neo4j driver is not installed. "
                        "Install with: pip install spiderweb[neo4j]"
                    ) from e
            self._document_processor = DocumentProcessor(
                llm_client=self.llm_client,
                chunker_config=self.chunker_config,
                validator_config=self.validator_config,
                store_config=self.store_config,
                extractor=self.extractor,
                chunk_add_ons=self.chunk_add_ons,
                chunk_addon_config=self.chunk_addon_config,
                graph_store=graph_store,
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

    async def ingest(
        self,
        file_path: str | Path,
        document_id: str | None = None,
    ) -> IngestionResult:
        """Ingest a single document.

        Args:
            file_path: Path to the document
            document_id: Optional custom document ID. If provided, this ID will be used
                for the document and all its chunks (useful for linking to external systems).
                If not provided, a UUID will be auto-generated.

        Returns:
            Ingestion result with statistics

        Raises:
            FileNotFoundError: If file doesn't exist
            ExtractionError: If extraction fails
        """
        return await self.document_processor.process(file_path, document_id=document_id)

    async def ingest_with_adaptive_chunking(
        self,
        file_path: str | Path,
        document_id: str | None = None,
        preview_max_chars: int = 4000,
    ) -> IngestionResult:
        """Ingest a document with adaptive chunking strategy selection.
        
        Uses an LLM agent to analyze the document preview and automatically
        select the best chunking strategy, then ingests the document with that strategy.
        
        Args:
            file_path: Path to the document
            document_id: Optional custom document ID. If provided, this ID will be used
                for the document and all its chunks (useful for linking to external systems).
                If not provided, a UUID will be auto-generated.
            preview_max_chars: Maximum number of characters to use for document preview
                when selecting chunking strategy (default: 4000)
        
        Returns:
            Ingestion result with statistics
        
        Raises:
            ValueError: If llm_client is not provided
            FileNotFoundError: If file doesn't exist
            ExtractionError: If extraction fails
        
        Example:
            >>> from gluellm import GlueLLM
            >>> from spiderweb import Spiderweb
            >>> 
            >>> llm = GlueLLM()
            >>> web = Spiderweb(llm_client=llm)
            >>> result = await web.ingest_with_adaptive_chunking("document.md")
            >>> print(f"Created {result.chunks_created} chunks")
        """
        if not self.llm_client:
            raise ValueError(
                "llm_client is required for adaptive chunking. "
                "Provide llm_client when initializing Spiderweb."
            )
        
        from pathlib import Path
        from spiderweb.chunking_agent import choose_chunking_strategy, create_chunker_from_strategy
        
        path = Path(file_path)
        
        # 1. Load document to get preview
        logger.info(f"Loading document preview for adaptive chunking: {path.name}")
        document = await self.document_processor.file_loader.load(path)
        
        # Override document ID if provided
        if document_id:
            document.id = document_id
        
        # 2. Build preview (use markdown_content if available, otherwise raw_content)
        preview_text = document.markdown_content[:preview_max_chars] if document.markdown_content else document.raw_content[:preview_max_chars]
        
        # 3. Choose chunking strategy using LLM agent
        logger.info(f"Analyzing document to select chunking strategy: {path.name}")
        choice = await choose_chunking_strategy(
            llm_client=self.llm_client,
            document_preview=preview_text,
            file_name=path.name,
        )
        
        logger.info(
            f"Selected chunking strategy '{choice.strategy}' for {path.name}: {choice.rationale}"
        )
        
        # 4. Create chunker from selected strategy
        chunker = create_chunker_from_strategy(
            strategy=choice.strategy,
            base_config=self.chunker_config,
            llm_client=self.llm_client,
        )
        
        # 5. Process document with the selected chunker
        return await self.document_processor.process(
            file_path=file_path,
            document_id=document_id,
            chunker_override=chunker,
        )

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

    async def delete_document(self, document_id: str) -> int:
        """Delete a document and all its chunks from the vector store.

        Args:
            document_id: The document ID to delete

        Returns:
            Number of chunks deleted
        """
        deleted_count = await self.document_processor.vector_store.delete_by_document_id(document_id)
        logger.info(f"Deleted document {document_id} with {deleted_count} chunks")
        return deleted_count

    async def get_document_chunks(self, document_id: str, limit: int = 1000) -> list:
        """Get all chunks for a document from the vector store.

        Args:
            document_id: The document ID to get chunks for
            limit: Maximum number of chunks to return

        Returns:
            List of Chunk objects sorted by chunk_index
        """
        chunks = await self.document_processor.vector_store.get_by_document_id(document_id, limit=limit)
        logger.debug(f"Retrieved {len(chunks)} chunks for document {document_id}")
        return chunks

    async def query(
        self,
        query: str,
        top_k: int = 10,
        filter_dict: dict | None = None,
        context_window: ContextWindowConfig | None = None,
        query_expansion: QueryExpansionConfig | None = None,
        rerank_config: RerankConfig | None = None,
        hybrid_config: HybridConfig | None = None,
    ) -> QueryResult | QueryResultWithContext:
        """Query the vector store.

        Args:
            query: Query string
            top_k: Number of results to return
            filter_dict: Optional metadata filters
            context_window: Optional context window configuration for surrounding chunks
            query_expansion: Optional query expansion configuration for improved recall
            rerank_config: Optional re-ranking configuration for improved precision
            hybrid_config: Optional hybrid search configuration (BM25 + vector)

        Returns:
            Query result with matched chunks, optionally with context

        Raises:
            ValueError: If no LLM client provided
        """
        if not self.llm_client:
            raise ValueError("LLM client required for querying. Provide llm_client when initializing Spiderweb.")

        import time

        from spiderweb.observability.tracing import span

        start_time = time.time()

        with span("query", {"query": query[:100], "top_k": top_k}):
            # Handle hybrid search if enabled (takes precedence over standard vector search)
            if hybrid_config and hybrid_config.enabled:
                from spiderweb.pipeline.hybrid import HybridSearcher

                logger.info("Hybrid search enabled (BM25 + vector)")
                searcher = HybridSearcher(
                    llm_client=self.llm_client,
                    vector_store=self.document_processor.vector_store,
                )
                results = await searcher.search(
                    query,
                    top_k=top_k,
                    filter_dict=filter_dict,
                    rrf_k=hybrid_config.rrf_k,
                )
            # Handle query expansion if enabled
            elif query_expansion and query_expansion.enabled:
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

            # Apply re-ranking if enabled
            if rerank_config and rerank_config.enabled:
                from spiderweb.pipeline.rerank import Reranker

                logger.info(f"Re-ranking enabled with model: {rerank_config.model}")
                reranker = Reranker(
                    llm_client=self.llm_client,
                    model=rerank_config.model,
                    model_name=rerank_config.model_name,
                )
                results = await reranker.rerank(query, results, top_k=rerank_config.top_k or top_k)

            execution_time = time.time() - start_time

            # Record metrics
            try:
                from spiderweb.observability.metrics import record_query_duration

                record_query_duration(execution_time)
            except Exception:
                pass

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
            expanded_queries = None
            expansion_strategy = None
            rrf_scores_list = None
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

    async def crawl_one(
        self,
        url: str,
        crawler_config: CrawlerConfig | None = None,
        extraction_config: CrawlExtractionConfig | None = None,
        output_schema: type | None = None,
        save_to: str | Path | None = None,
        save_format: str = "all",
    ) -> CrawlResult:
        """Crawl a single URL and return a single `CrawlResult`.

        Prefer this over `crawl()` when you want a stable return type.

        Raises:
            ValueError: If crawling returns multiple pages (e.g. max_depth > 1).
        """
        result = await self.crawl(
            url=url,
            crawler_config=crawler_config,
            extraction_config=extraction_config,
            output_schema=output_schema,
            ingest=False,
            save_to=save_to,
            save_format=save_format,
        )
        if isinstance(result, list):
            raise ValueError("crawl_one() got multiple results. Use crawl_many() or set crawler_config.max_depth=1.")
        if not isinstance(result, CrawlResult):
            raise TypeError(f"crawl_one() expected CrawlResult, got {type(result).__name__}")
        return result

    async def crawl_many(
        self,
        urls: list[str],
        crawler_config: CrawlerConfig | None = None,
        extraction_config: CrawlExtractionConfig | None = None,
        output_schema: type | None = None,
        save_to: str | Path | None = None,
        save_format: str = "all",
    ) -> list[CrawlResult]:
        """Crawl multiple URLs and return a list of `CrawlResult`."""
        result = await self.crawl(
            url=urls,
            crawler_config=crawler_config,
            extraction_config=extraction_config,
            output_schema=output_schema,
            ingest=False,
            save_to=save_to,
            save_format=save_format,
        )
        if not isinstance(result, list):
            raise TypeError(f"crawl_many() expected list[CrawlResult], got {type(result).__name__}")
        return result

    async def crawl_and_ingest_one(
        self,
        url: str,
        crawler_config: CrawlerConfig | None = None,
        extraction_config: CrawlExtractionConfig | None = None,
        output_schema: type | None = None,
    ) -> IngestionResult:
        """Crawl a single URL and ingest it, returning an `IngestionResult`.

        Raises:
            ValueError: If crawling returns multiple pages (e.g. max_depth > 1).
        """
        result = await self.crawl(
            url=url,
            crawler_config=crawler_config,
            extraction_config=extraction_config,
            output_schema=output_schema,
            ingest=True,
        )
        if isinstance(result, BatchIngestionResult):
            raise ValueError(
                "crawl_and_ingest_one() got a batch result. Use crawl_and_ingest_many() "
                "or set crawler_config.max_depth=1."
            )
        if not isinstance(result, IngestionResult):
            raise TypeError(f"crawl_and_ingest_one() expected IngestionResult, got {type(result).__name__}")
        return result

    async def crawl_and_ingest_many(
        self,
        urls: list[str],
        crawler_config: CrawlerConfig | None = None,
        extraction_config: CrawlExtractionConfig | None = None,
        output_schema: type | None = None,
    ) -> BatchIngestionResult:
        """Crawl multiple URLs and ingest them, returning a `BatchIngestionResult`."""
        result = await self.crawl(
            url=urls,
            crawler_config=crawler_config,
            extraction_config=extraction_config,
            output_schema=output_schema,
            ingest=True,
        )
        if not isinstance(result, BatchIngestionResult):
            raise TypeError(f"crawl_and_ingest_many() expected BatchIngestionResult, got {type(result).__name__}")
        return result

    @overload
    async def crawl(
        self,
        url: str,
        crawler_config: CrawlerConfig | None = None,
        extraction_config: CrawlExtractionConfig | None = None,
        output_schema: type | None = None,
        ingest: Literal[False] = False,
        save_to: str | Path | None = None,
        save_format: str = "all",
    ) -> CrawlResult: ...

    @overload
    async def crawl(
        self,
        url: list[str],
        crawler_config: CrawlerConfig | None = None,
        extraction_config: CrawlExtractionConfig | None = None,
        output_schema: type | None = None,
        ingest: Literal[False] = False,
        save_to: str | Path | None = None,
        save_format: str = "all",
    ) -> list[CrawlResult]: ...

    @overload
    async def crawl(
        self,
        url: str,
        crawler_config: CrawlerConfig | None = None,
        extraction_config: CrawlExtractionConfig | None = None,
        output_schema: type | None = None,
        ingest: Literal[True] = True,
        save_to: str | Path | None = None,
        save_format: str = "all",
    ) -> IngestionResult: ...

    @overload
    async def crawl(
        self,
        url: list[str],
        crawler_config: CrawlerConfig | None = None,
        extraction_config: CrawlExtractionConfig | None = None,
        output_schema: type | None = None,
        ingest: Literal[True] = True,
        save_to: str | Path | None = None,
        save_format: str = "all",
    ) -> BatchIngestionResult: ...

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
        
        # Discover URLs from sitemap if enabled
        if crawler_config and crawler_config.use_sitemap:
            from spiderweb.crawlers.sitemap import discover_sitemap_url, parse_sitemap
            import aiohttp
            
            sitemap_urls = []
            async with aiohttp.ClientSession() as session:
                for base_url in urls:
                    sitemap_url = discover_sitemap_url(base_url)
                    try:
                        discovered = await parse_sitemap(sitemap_url, session)
                        sitemap_urls.extend(discovered)
                        logger.info(f"Discovered {len(discovered)} URLs from sitemap: {sitemap_url}")
                    except Exception as e:
                        logger.warning(f"Failed to parse sitemap {sitemap_url}: {e}")
            
            if sitemap_urls:
                # Merge discovered URLs with original URLs (deduplicate)
                all_urls = list(set(urls + sitemap_urls))
                logger.info(f"Merged {len(sitemap_urls)} sitemap URLs with {len(urls)} original URLs")
                urls = all_urls
        
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
                
                # Ingest each document via single pipeline path
                results = []
                for doc in documents:
                    result = await self.document_processor.process_document(
                        doc, store_chunks=True
                    )
                    results.append(result)

                successful = sum(1 for r in results if r.success)
                errors_by_source = {
                    doc.metadata.source: r.errors
                    for doc, r in zip(documents, results, strict=True)
                    if r.errors
                }
                return BatchIngestionResult(
                    total_documents=len(documents),
                    successful_documents=successful,
                    failed_documents=len(results) - successful,
                    total_chunks=sum(r.chunks_created for r in results),
                    total_chunks_validated=sum(r.chunks_validated for r in results),
                    total_chunks_rejected=sum(r.chunks_rejected for r in results),
                    processing_time_seconds=sum(r.processing_time_seconds for r in results),
                    average_time_per_document=sum(r.processing_time_seconds for r in results) / len(results) if results else 0.0,
                    results=results,
                    errors=errors_by_source,
                )
            
            else:
                # Single document
                doc = await web_loader.load(
                    urls[0],
                    crawler_config=crawler_config,
                    extraction_config=extraction_config,
                    output_schema=output_schema,
                )
                return await self.document_processor.process_document(
                    doc, store_chunks=True
                )

    async def search_crawl_extract(
        self,
        query: str,
        search_provider_config: SearchProviderConfig | None = None,
        crawler_config: CrawlerConfig | None = None,
        extraction_config: CrawlExtractionConfig | None = None,
        depth_config: SearchDepthConfig | None = None,
        output_schema: type | None = None,
        ingest: bool = False,
        save_to: str | Path | None = None,
        save_format: str | None = None,
        save_trace_to: str | Path | None = None,
        trace_format: Literal["json", "markdown", "jsonl"] | None = None,
    ) -> SearchCrawlTrace:
        """Search the web, crawl results, and optionally extract structured data.
        
        Implements a multi-round search → crawl → extract pipeline with configurable
        depth, relevance filtering, and query expansion. Builds a complete trace
        of the search session including summaries, URLs, and provenance.
        
        Args:
            query: Initial search query
            search_provider_config: Search provider configuration
            crawler_config: Crawler configuration (includes relevance prompt)
            extraction_config: Optional extraction configuration
            depth_config: Configuration for multi-round search and "go deeper" strategy
            output_schema: Optional Pydantic schema for structured extraction
            ingest: If True, ingest crawled content into vector store
            save_to: Optional directory to save crawled content locally
            save_format: Format for saved files - "markdown", "html", "json", or "all"
            save_trace_to: Optional path to save search-crawl trace
            trace_format: Format for trace file (json, markdown, or jsonl)
            
        Returns:
            SearchCrawlTrace containing full session flow
            
        Example:
            >>> from spiderweb.models.config import SearchProviderConfig, SearchDepthConfig
            >>> 
            >>> trace = await web.search_crawl_extract(
            ...     "python web scraping",
            ...     search_provider_config=SearchProviderConfig(provider="stub", limit=5),
            ...     depth_config=SearchDepthConfig(max_search_rounds=2),
            ...     save_trace_to="./trace.json",
            ... )
            >>> print(f"Crawled {len(trace.get_all_urls())} URLs")
        """
        # Initialize configs
        search_config = search_provider_config or SearchProviderConfig()
        crawler_config = crawler_config or CrawlerConfig()
        depth_config = depth_config or SearchDepthConfig()
        
        # Use settings defaults if not provided
        if save_format is None:
            save_format = settings.default_save_format
        if trace_format is None:
            trace_format = settings.default_trace_format
        
        # Create trace
        trace = SearchCrawlTrace(
            original_query=query,
            config_snapshot={
                "search_provider": search_config.provider,
                "max_rounds": depth_config.max_search_rounds,
                "crawl_per_round": depth_config.crawl_results_per_round,
                "when_to_go_deeper": depth_config.when_to_go_deeper,
            },
        )
        
        # Initialize search provider
        try:
            search_provider = search_provider_registry.create(
                search_config.provider,
                **search_config.extra_config,
            )
        except KeyError as e:
            available_search = search_provider_registry.list()
            crawler_names = crawler_registry.list()
            hint = ""
            if search_config.provider in crawler_names:
                hint = (
                    f" '{search_config.provider}' is a crawler (fetches URLs), not a search provider (finds URLs). "
                    f"Use --crawl-provider {search_config.provider} for crawling. "
                )
            raise ValueError(
                f"Unknown search provider: {search_config.provider}. "
                f"Available search providers: {available_search}.{hint}"
            ) from e
        
        # Initialize web loader for crawling
        web_loader = WebLoader(
            llm_client=self.llm_client,
            crawler_config=crawler_config,
            extraction_config=extraction_config,
        )
        
        # Initialize storage if needed (by query so each search has its own subdir)
        storage = None
        if save_to:
            from spiderweb.crawlers.storage import CrawlStorage
            query_slug = sanitize_query_for_path(query)
            output_dir = Path(save_to) / query_slug
            storage = CrawlStorage(output_dir=output_dir)
            logger.info(f"Will save crawled content to {output_dir}")
        
        # Track total pages crawled
        total_pages_crawled = 0
        all_crawled_urls = set()
        
        # Multi-round search loop
        queries_to_try = [query]
        
        for round_num in range(1, depth_config.max_search_rounds + 1):
            logger.info(f"Search round {round_num}/{depth_config.max_search_rounds}")
            
            # Check if we've hit max pages limit
            if depth_config.max_pages_total and total_pages_crawled >= depth_config.max_pages_total:
                logger.info(f"Reached max_pages_total limit ({depth_config.max_pages_total})")
                break
            
            # Generate expanded queries if needed
            if round_num > 1 and depth_config.when_to_go_deeper == "expand_queries":
                if self.llm_client:
                    from spiderweb.pipeline.query_expansion import QueryExpander
                    from spiderweb.models.config import QueryExpansionConfig
                    
                    expander = QueryExpander(
                        self.llm_client,
                        QueryExpansionConfig(
                            strategy="multi_query",
                            num_expansions=depth_config.num_expanded_queries,
                            include_original=False,
                        ),
                    )
                    expanded = await expander.expand(query)
                    queries_to_try = expanded[:depth_config.num_expanded_queries]
                    logger.info(f"Generated {len(queries_to_try)} expanded queries for round {round_num}")
                else:
                    # No LLM, can't expand - use original query
                    queries_to_try = [query]
            elif round_num == 1:
                # First round: use original query
                queries_to_try = [query]
            
            # Set current_query for this round (used in trace)
            current_query = queries_to_try[0] if queries_to_try else query
            
            # Execute searches for this round
            round_pages = []
            round_filtered = []
            round_search_results = []
            
            for search_query in queries_to_try:
                logger.info(f"  Searching: {search_query}")
                # Search
                search_results = await search_provider.search(
                    search_query,
                    limit=search_config.limit,
                    **search_config.extra_config,
                )
                round_search_results.extend(search_results.results)
            
            # Dedupe URLs
            seen_urls = set()
            unique_results = []
            for result in round_search_results:
                if result.url not in seen_urls and result.url not in all_crawled_urls:
                    seen_urls.add(result.url)
                    unique_results.append(result)
            
            # Apply relevance filter if configured
            candidates_to_crawl = unique_results
            if crawler_config.crawl_relevance_prompt:
                good_candidates, bad_candidates = await filter_by_relevance(
                    self.llm_client,
                    unique_results,
                    crawler_config.crawl_relevance_prompt,
                    use_llm=crawler_config.crawl_relevance_use_llm,
                )
                candidates_to_crawl = good_candidates
                
                # Record filtered candidates
                for bad in bad_candidates:
                    if isinstance(bad, FilteredCandidate):
                        bad.source_query = current_query
                        round_filtered.append(bad)
                    else:
                        # Dict case
                        round_filtered.append(
                            FilteredCandidate(
                                url=bad.get("url", ""),
                                title=bad.get("title"),
                                snippet=bad.get("snippet"),
                                source_query=current_query,
                                filter_reason=bad.get("filter_reason", "relevance"),
                            )
                        )
            
            # Limit to crawl_results_per_round and respect max_pages_total
            remaining_slots = depth_config.crawl_results_per_round
            if depth_config.max_pages_total:
                remaining_slots = min(
                    remaining_slots,
                    depth_config.max_pages_total - total_pages_crawled,
                )
            
            candidates_to_crawl = candidates_to_crawl[:remaining_slots]
            
            if not candidates_to_crawl:
                logger.info("No candidates to crawl in this round")
                # Still record the round with filtered results
                round_data = SearchRound(
                    query=current_query,
                    search_results=SearchResultBatch(
                        results=round_search_results if round_search_results else [],
                        query=current_query,
                        total=len(round_search_results),
                    ),
                    pages=[],
                    filtered_out=round_filtered,
                )
                trace.add_round(round_data)
                break
            
            # Extract URLs and drop candidates with empty/missing URL so crawler never sees invalid URLs
            valid_pairs = []
            for r in candidates_to_crawl:
                u = (r.url if isinstance(r, SearchResult) else r.get("url")) or ""
                u = (u or "").strip()
                if u:
                    valid_pairs.append((r, u))
            candidates_to_crawl = [c for c, _ in valid_pairs]
            urls_to_crawl = [u for _, u in valid_pairs]
            if not urls_to_crawl:
                logger.info("No valid URLs to crawl in this round (all empty or missing)")
                round_data = SearchRound(
                    query=current_query,
                    search_results=SearchResultBatch(
                        results=round_search_results,
                        query=current_query,
                        total=len(round_search_results),
                    ),
                    pages=[],
                    filtered_out=round_filtered,
                )
                trace.add_round(round_data)
                break

            # Crawl
            logger.info(f"Crawling {len(urls_to_crawl)} URLs")
            crawl_results = await web_loader.crawler.crawl_many(urls_to_crawl, crawler_config)
            
            # Build PageRecords
            for i, crawl_result in enumerate(crawl_results):
                if not isinstance(crawl_result, CrawlResult):
                    continue
                
                candidate = candidates_to_crawl[i]
                url = candidate.url if isinstance(candidate, SearchResult) else candidate.get("url", "")
                
                # Generate summary (truncate or use LLM if available)
                summary = None
                if crawl_result.markdown:
                    summary = crawl_result.markdown[:500] + "..." if len(crawl_result.markdown) > 500 else crawl_result.markdown
                
                # Extract structured data if configured
                extracted_data = None
                if extraction_config and extraction_config.enabled and self.llm_client:
                    from spiderweb.crawlers.extraction import CrawlExtractor
                    
                    extractor = CrawlExtractor(self.llm_client, extraction_config)
                    try:
                        extracted_data = await extractor.extract(
                            crawl_result.markdown or crawl_result.content,
                            schema=output_schema or extraction_config.output_schema,
                            semantic_guide=extraction_config.semantic_guide,
                            extraction_query=extraction_config.extraction_query,
                        )
                    except Exception as e:
                        logger.warning(f"Extraction failed for {url}: {e}")
                
                page_record = PageRecord(
                    url=url,
                    summary=summary,
                    crawl_result=crawl_result,
                    source_query=current_query,
                    source_position=(
                        candidate.position
                        if isinstance(candidate, SearchResult)
                        else None
                    ),
                    extracted_data=extracted_data,
                    links_found=crawl_result.links,
                )
                round_pages.append(page_record)
                all_crawled_urls.add(url)
                total_pages_crawled += 1
                
                # Save to local storage if requested
                if storage:
                    storage.save_crawl_result(crawl_result, format=save_format)
            
            # Create SearchRound and add to trace
            round_data = SearchRound(
                query=current_query,
                search_results=SearchResultBatch(
                    results=round_search_results,
                    query=current_query,
                    total=len(round_search_results),
                ),
                pages=round_pages,
                filtered_out=round_filtered,
            )
            trace.add_round(round_data)
            
            # Decide if we should continue
            if round_num >= depth_config.max_search_rounds:
                break
            
            if depth_config.when_to_go_deeper == "always":
                # Continue to next round
                continue
            elif depth_config.when_to_go_deeper == "if_not_found":
                # Check if answer found (simplified: check if we got good results)
                if len(round_pages) >= depth_config.crawl_results_per_round // 2:
                    logger.info("Sufficient results found, stopping")
                    break
        
        # Save trace if requested (by query so each search has its own file)
        if save_trace_to:
            base = Path(save_trace_to)
            query_slug = sanitize_query_for_path(query)
            if base.suffix.lower() in (".json", ".jsonl", ".md"):
                trace_dir = base.parent
            else:
                trace_dir = base
            trace_dir.mkdir(parents=True, exist_ok=True)
            trace_path = trace_dir / f"{query_slug}.{trace_format}"
            write_trace(trace, trace_path, format=trace_format)
            logger.info(f"Saved trace to {trace_path}")
        
        # Create index if storage was used
        if storage:
            storage.create_index()
        
        # Handle ingestion if requested
        if ingest:
            for round_data in trace.rounds:
                for page in round_data.pages:
                    if page.crawl_result and page.crawl_result.success:
                        doc = await web_loader.load(
                            page.url,
                            crawler_config=crawler_config,
                            extraction_config=extraction_config,
                            output_schema=output_schema,
                        )
                        await self.document_processor.process_document(
                            doc, store_chunks=True
                        )
        
        return trace

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
        """Async context manager exit. Closes graph store and vector store if they support close()."""
        import asyncio

        if self._document_processor is None:
            return

        proc = self._document_processor
        if proc.graph_store is not None:
            close_fn = getattr(proc.graph_store, "close", None)
            if close_fn is not None:
                try:
                    if asyncio.iscoroutinefunction(close_fn):
                        await close_fn()
                    else:
                        close_fn()
                except Exception as e:
                    logger.warning("Error closing graph store: %s", e)

        if getattr(proc, "vector_store", None) is not None:
            close_fn = getattr(proc.vector_store, "close", None)
            if close_fn is not None:
                try:
                    if asyncio.iscoroutinefunction(close_fn):
                        await close_fn()
                    else:
                        close_fn()
                except Exception as e:
                    logger.warning("Error closing vector store: %s", e)


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

"""Spiderweb API - High-level interface for document processing and RAG.

This module provides the main API functions for working with Spiderweb,
following the same patterns as gluellm for a consistent user experience.
"""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gluellm import GlueLLM

    from spiderweb.extractors.base import Extractor

from spiderweb.models.config import ChunkerConfig, ValidatorConfig, VectorStoreConfig
from spiderweb.models.document import Document
from spiderweb.models.result import BatchIngestionResult, IngestionResult, QueryResult
from spiderweb.observability.logging_config import get_logger
from spiderweb.pipeline.batch import BatchProcessor
from spiderweb.pipeline.processor import DocumentProcessor

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
    ) -> QueryResult:
        """Query the vector store.

        Args:
            query: Query string
            top_k: Number of results to return
            filter_dict: Optional metadata filters

        Returns:
            Query result with matched chunks

        Raises:
            ValueError: If no LLM client provided
        """
        if not self.llm_client:
            raise ValueError("LLM client required for querying. Provide llm_client when initializing Spiderweb.")

        import time

        start_time = time.time()

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

        return QueryResult(
            query=query,
            chunks=chunks,
            scores=scores,
            execution_time_seconds=execution_time,
            total_results=len(chunks),
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
) -> QueryResult:
    """Quick vector store query.

    Args:
        query_text: Query string
        llm_client: GlueLLM client for embeddings
        vector_store_url: Vector store connection URL
        top_k: Number of results

    Returns:
        Query result

    Example:
        >>> from gluellm import GlueLLM
        >>> from spiderweb import query
        >>>
        >>> results = await query(
        ...     "What is machine learning?",
        ...     llm_client=GlueLLM(),
        ...     top_k=5,
        ... )
        >>> for chunk in results.chunks:
        ...     print(chunk["content"])
    """
    web = Spiderweb(
        llm_client=llm_client,
        vector_store_url=vector_store_url,
    )

    return await web.query(query_text, top_k=top_k)

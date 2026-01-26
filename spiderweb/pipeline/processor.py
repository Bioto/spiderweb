"""Main document processing pipeline.

Orchestrates extraction, chunking, validation, embedding, and storage.
"""

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gluellm import GlueLLM

    from spiderweb.chunkers.base import Chunker
    from spiderweb.extractors.base import Extractor
    from spiderweb.stores.base import VectorStore
    from spiderweb.validators.pipeline import ValidationPipeline

from spiderweb.extractors.markitdown import MarkitdownExtractor
from spiderweb.hooks import HookManager, HookPoint, hooks as global_hooks
from spiderweb.loaders.file_loader import FileLoader
from spiderweb.models.config import ChunkerConfig, ValidatorConfig, VectorStoreConfig
from spiderweb.models.document import Chunk, Document, DocumentMetadata
from spiderweb.models.result import IngestionResult
from spiderweb.observability.logging_config import get_logger
from spiderweb.registry import chunker_registry
from spiderweb.stores.memory import MemoryVectorStore
from spiderweb.validators.pipeline import ValidationPipeline

logger = get_logger(__name__)


class DocumentProcessor:
    """Main pipeline for processing documents.

    Orchestrates the full pipeline: extraction → chunking → validation → embedding → storage

    Example:
        >>> from gluellm import GlueLLM
        >>> from spiderweb.pipeline import DocumentProcessor
        >>>
        >>> llm = GlueLLM()
        >>> processor = DocumentProcessor(llm_client=llm)
        >>> result = await processor.process("document.pdf")
        >>> print(f"Created {result.chunks_created} chunks")
    """

    def __init__(
        self,
        llm_client: "GlueLLM | None" = None,
        extractor: "Extractor | None" = None,
        chunker: "Chunker | None" = None,
        validator: "ValidationPipeline | None" = None,
        vector_store: "VectorStore | None" = None,
        chunker_config: ChunkerConfig | None = None,
        validator_config: ValidatorConfig | None = None,
        store_config: VectorStoreConfig | None = None,
        enable_validation: bool = True,
        enable_embedding: bool = True,
        hook_manager: HookManager | None = None,
    ):
        """Initialize document processor.

        Args:
            llm_client: GlueLLM client for embeddings
            extractor: Document extractor (defaults to MarkitdownExtractor)
            chunker: Chunking strategy (defaults to HierarchicalChunker)
            validator: Validation pipeline
            vector_store: Vector store (defaults to MemoryVectorStore)
            chunker_config: Chunker configuration
            validator_config: Validator configuration
            store_config: Vector store configuration
            enable_validation: Enable chunk validation
            enable_embedding: Enable embedding generation
            hook_manager: Optional hook manager for pipeline hooks (defaults to global hooks)
        """
        self.llm_client = llm_client
        self.enable_validation = enable_validation
        self.enable_embedding = enable_embedding
        self.hooks = hook_manager or global_hooks

        # Initialize components
        self.extractor = extractor or MarkitdownExtractor()
        self.file_loader = FileLoader(extractor=self.extractor)

        # Initialize chunker via registry
        if chunker:
            self.chunker = chunker
        else:
            config = chunker_config or ChunkerConfig()
            # Get strategy name from config (handles both enum and string)
            strategy = config.strategy.value if hasattr(config.strategy, "value") else str(config.strategy)

            # Look up chunker class in registry
            if strategy in chunker_registry:
                chunker_cls = chunker_registry.get(strategy)
                # Use from_config if available, otherwise direct instantiation
                if hasattr(chunker_cls, "from_config"):
                    self.chunker = chunker_cls.from_config(config)
                else:
                    self.chunker = chunker_cls()
            else:
                # Fallback to hierarchical for unknown strategies
                logger.warning(
                    f"Unknown chunker strategy '{strategy}', falling back to hierarchical. "
                    f"Available: {chunker_registry.list()}"
                )
                from spiderweb.chunkers.hierarchical import HierarchicalChunker
                self.chunker = HierarchicalChunker.from_config(config)

        # Initialize validator
        if enable_validation:
            if validator:
                self.validator = validator
            else:
                config = validator_config or ValidatorConfig()
                self.validator = ValidationPipeline(llm_client=llm_client, config=config)
        else:
            self.validator = None

        # Initialize vector store
        if vector_store:
            self.vector_store = vector_store
        elif store_config:
            # Initialize store from config
            if store_config.provider == "qdrant":
                from spiderweb.stores.qdrant import QdrantVectorStore

                self.vector_store = QdrantVectorStore.from_config(store_config)
                logger.info(f"Initialized Qdrant vector store: {store_config.collection_name}")
            else:
                # Default to memory store
                self.vector_store = MemoryVectorStore()
        else:
            # Default to memory store
            self.vector_store = MemoryVectorStore()

        logger.info(
            f"Initialized DocumentProcessor: "
            f"extractor={type(self.extractor).__name__}, "
            f"chunker={type(self.chunker).__name__}, "
            f"vector_store={type(self.vector_store).__name__}, "
            f"validation={enable_validation}, "
            f"embedding={enable_embedding}"
        )

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Uses a simple heuristic: ~4 characters per token on average.

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        return len(text) // 4

    def _create_embedding_batches(self, chunks: list[Chunk]) -> list[list[Chunk]]:
        """Create batches of chunks that fit within token limits.

        Args:
            chunks: Chunks to batch

        Returns:
            List of chunk batches
        """
        from spiderweb.config import settings

        max_tokens = settings.max_tokens_per_embedding_batch
        max_batch_size = settings.embedding_batch_size

        batches = []
        current_batch = []
        current_tokens = 0

        for chunk in chunks:
            chunk_tokens = self._estimate_tokens(chunk.content)

            # If this single chunk exceeds limit, we need to split it or handle specially
            if chunk_tokens > max_tokens:
                logger.warning(
                    f"Chunk {chunk.id} has {chunk_tokens} estimated tokens, "
                    f"exceeds max {max_tokens}. Will attempt to embed anyway."
                )
                # Add current batch if not empty
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_tokens = 0
                # Add oversized chunk as its own batch
                batches.append([chunk])
                continue

            # Check if adding this chunk would exceed limits
            if (current_tokens + chunk_tokens > max_tokens) or (len(current_batch) >= max_batch_size):
                # Start new batch
                if current_batch:
                    batches.append(current_batch)
                current_batch = [chunk]
                current_tokens = chunk_tokens
            else:
                # Add to current batch
                current_batch.append(chunk)
                current_tokens += chunk_tokens

        # Add remaining batch
        if current_batch:
            batches.append(current_batch)

        return batches

    async def _embed_batch(self, batch: list[Chunk], batch_num: int, total_batches: int) -> list[Chunk]:
        """Embed a single batch of chunks.

        Args:
            batch: Chunks to embed
            batch_num: Batch number (for logging)
            total_batches: Total number of batches

        Returns:
            Chunks with embeddings
        """
        texts = [chunk.content for chunk in batch]
        estimated_tokens = sum(self._estimate_tokens(t) for t in texts)

        logger.debug(f"Embedding batch {batch_num}/{total_batches}: {len(texts)} chunks, ~{estimated_tokens} tokens")

        try:
            embedding_result = await self.llm_client.embed(texts)

            # Assign embeddings to chunks
            for chunk, embedding in zip(batch, embedding_result.embeddings, strict=True):
                chunk.embedding = embedding

            return batch

        except Exception as e:
            logger.error(f"Failed to embed batch {batch_num}/{total_batches}: {e}. Chunks will have no embeddings.")
            # Return chunks without embeddings rather than failing completely
            return batch

    async def _generate_embeddings(self, chunks: list[Chunk]) -> list[Chunk]:
        """Generate embeddings for chunks with batching and parallelization.

        Automatically batches chunks to stay within token limits and processes
        batches concurrently for better performance.

        Args:
            chunks: Chunks to embed

        Returns:
            Chunks with embeddings
        """
        if not self.llm_client:
            logger.warning("No LLM client provided, skipping embedding generation")
            return chunks

        if not chunks:
            return chunks

        start_time = time.time()

        # Create batches that fit within token limits
        batches = self._create_embedding_batches(chunks)

        logger.info(
            f"Generating embeddings for {len(chunks)} chunks in {len(batches)} batches "
            f"(max concurrent: {self.llm_client.embedding_model or 'default'})"
        )

        # Process batches concurrently with semaphore for rate limiting
        from spiderweb.config import settings

        semaphore = asyncio.Semaphore(settings.max_concurrent_embeddings)

        async def process_batch_with_limit(batch: list[Chunk], batch_num: int) -> list[Chunk]:
            async with semaphore:
                return await self._embed_batch(batch, batch_num, len(batches))

        # Process all batches concurrently
        tasks = [process_batch_with_limit(batch, i + 1) for i, batch in enumerate(batches)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch {i + 1} failed with exception: {result}")

        elapsed = time.time() - start_time
        chunks_with_embeddings = sum(1 for chunk in chunks if chunk.embedding is not None)

        logger.info(
            f"Generated {chunks_with_embeddings}/{len(chunks)} embeddings in {elapsed:.2f}s "
            f"({elapsed / len(batches):.2f}s per batch avg)"
        )

        return chunks

    def _create_skipped_result(
        self,
        path: Path,
        start_time: float,
        reason: str,
    ) -> IngestionResult:
        """Create an IngestionResult for a skipped document.

        Args:
            path: Path to the document
            start_time: Processing start time
            reason: Reason for skipping

        Returns:
            IngestionResult with success=True but no chunks
        """
        processing_time = time.time() - start_time
        return IngestionResult(
            document=Document(
                raw_content="",
                markdown_content="",
                metadata=DocumentMetadata(
                    source=str(path),
                    file_type=path.suffix.lstrip(".") or "unknown",
                    extraction_method="skipped",
                ),
            ),
            success=True,
            chunks_created=0,
            chunks_validated=0,
            chunks_rejected=0,
            processing_time_seconds=processing_time,
            embedding_time_seconds=0.0,
            errors=[],
            warnings=[reason],
        )

    async def process(self, file_path: str | Path, store_chunks: bool = True) -> IngestionResult:
        """Process a single document through the full pipeline.

        Args:
            file_path: Path to the document
            store_chunks: Whether to store chunks in vector store

        Returns:
            Ingestion result with statistics

        Raises:
            FileNotFoundError: If file doesn't exist
            ExtractionError: If extraction fails
        """
        path = Path(file_path)
        start_time = time.time()

        logger.info(f"Processing document: {path.name}")

        errors = []
        warnings = []

        try:
            # Hook: BEFORE_EXTRACT
            ctx = await self.hooks.run(HookPoint.BEFORE_EXTRACT, str(path), file_path=str(path))
            if ctx.skip:
                logger.info(f"Extraction skipped by hook for {path.name}")
                return self._create_skipped_result(path, start_time, "Skipped by BEFORE_EXTRACT hook")
            extract_path = Path(ctx.modified_data) if ctx.modified_data else path

            # 1. Load and extract
            logger.debug(f"Step 1/5: Extracting content from {extract_path.name}")
            document = await self.file_loader.load(extract_path)

            # Hook: AFTER_EXTRACT
            ctx = await self.hooks.run(HookPoint.AFTER_EXTRACT, document, file_path=str(path))
            if ctx.modified_data is not None:
                document = ctx.modified_data

            # Hook: BEFORE_CHUNK
            ctx = await self.hooks.run(HookPoint.BEFORE_CHUNK, document, file_path=str(path))
            if ctx.skip:
                logger.info(f"Chunking skipped by hook for {path.name}")
                return self._create_skipped_result(path, start_time, "Skipped by BEFORE_CHUNK hook")
            if ctx.modified_data is not None:
                document = ctx.modified_data

            # 2. Chunk
            logger.debug("Step 2/5: Chunking document")
            chunks = self.chunker.chunk(document)

            # Hook: AFTER_CHUNK
            ctx = await self.hooks.run(HookPoint.AFTER_CHUNK, chunks, document=document, file_path=str(path))
            if ctx.modified_data is not None:
                chunks = ctx.modified_data

            document.chunks = chunks

            chunks_created = len(chunks)
            logger.info(f"Created {chunks_created} chunks from {path.name}")

            # 3. Validate
            chunks_validated = 0
            chunks_rejected = 0

            if self.enable_validation and self.validator:
                logger.debug(f"Step 3/5: Validating {len(chunks)} chunks")
                validation_results = await self.validator.validate_batch(chunks)

                # Filter out invalid chunks
                valid_chunks = []
                for chunk, result in zip(chunks, validation_results, strict=True):
                    if result.passed:
                        valid_chunks.append(chunk)
                        chunks_validated += 1
                    else:
                        chunks_rejected += 1
                        logger.debug(f"Rejected chunk {chunk.id}: {result.issues}")

                chunks = valid_chunks
                document.chunks = chunks

                if chunks_rejected > 0:
                    warnings.append(f"Rejected {chunks_rejected} low-quality chunks")
            else:
                chunks_validated = len(chunks)
                logger.debug("Step 3/5: Skipping validation (disabled)")

            # 4. Generate embeddings
            embedding_start = time.time()

            if self.enable_embedding and self.llm_client:
                logger.debug(f"Step 4/5: Generating embeddings for {len(chunks)} chunks")
                chunks = await self._generate_embeddings(chunks)
                document.chunks = chunks
            else:
                logger.debug("Step 4/5: Skipping embedding generation (disabled or no LLM client)")

            embedding_time = time.time() - embedding_start

            # 5. Store in vector database
            if store_chunks and chunks:
                logger.debug(f"Step 5/5: Storing {len(chunks)} chunks in vector store")

                # Only store chunks with embeddings
                chunks_with_embeddings = [c for c in chunks if c.embedding is not None]

                if chunks_with_embeddings:
                    await self.vector_store.upsert(chunks_with_embeddings)
                    logger.info(f"Stored {len(chunks_with_embeddings)} chunks in vector store")
                elif self.enable_embedding:
                    warnings.append("No chunks with embeddings to store")
            else:
                logger.debug("Step 5/5: Skipping vector store (disabled or no chunks)")

            processing_time = time.time() - start_time

            # Create result
            result = IngestionResult(
                document=document,
                success=True,
                chunks_created=chunks_created,
                chunks_validated=chunks_validated,
                chunks_rejected=chunks_rejected,
                chunks_deduplicated=0,  # Tracked by validator
                processing_time_seconds=processing_time,
                embedding_time_seconds=embedding_time,
                errors=errors,
                warnings=warnings,
            )

            logger.info(
                f"Successfully processed {path.name}: "
                f"{chunks_created} chunks, {chunks_validated} validated, "
                f"{processing_time:.2f}s"
            )

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Failed to process {path.name}: {e}"
            logger.error(error_msg, exc_info=True)
            errors.append(error_msg)

            # Return failed result
            failed_metadata = (
                document.metadata
                if "document" in locals() and hasattr(document, "metadata")
                else DocumentMetadata(
                    source=str(path),
                    file_type="unknown",
                    extraction_method="none",
                )
            )

            return IngestionResult(
                document=Document(
                    raw_content="",
                    markdown_content="",
                    metadata=failed_metadata,
                ),
                success=False,
                chunks_created=0,
                chunks_validated=0,
                chunks_rejected=0,
                processing_time_seconds=processing_time,
                embedding_time_seconds=0.0,
                errors=errors,
                warnings=warnings,
            )

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
    from spiderweb.stores.graph_base import GraphStore
    from spiderweb.validators.pipeline import ValidationPipeline

from spiderweb.extractors.markitdown import MarkitdownExtractor
from spiderweb.hooks import HookManager, HookPoint, hooks as global_hooks
from spiderweb.loaders.file_loader import FileLoader
from spiderweb.models.config import ChunkAddOnConfig, ChunkerConfig, ValidatorConfig, VectorStoreConfig
from spiderweb.models.document import Chunk, Document, DocumentMetadata
from spiderweb.models.result import IngestionResult
from spiderweb.observability.logging_config import get_logger
from spiderweb.pipeline.graph_adapter import (
    _scope_attributes_from_document,
    document_to_entities_and_relationships,
)
from spiderweb.registry import chunk_addon_registry, chunker_registry
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
        chunk_add_ons: list[str] | None = None,
        chunk_addon_config: ChunkAddOnConfig | None = None,
        graph_store: "GraphStore | None" = None,
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
            chunk_add_ons: List of add-on names to enable (e.g., ["facts"])
            chunk_addon_config: Chunk add-on configuration
            graph_store: Optional graph store for entities and relationships (e.g. Neo4j)
        """
        self.llm_client = llm_client
        self.graph_store = graph_store
        self.enable_validation = enable_validation
        self.enable_embedding = enable_embedding
        self.hooks = hook_manager or global_hooks
        
        # Store chunk add-on configuration
        self.chunk_add_ons = chunk_add_ons or []
        self.chunk_addon_config = chunk_addon_config or ChunkAddOnConfig()
        
        # If chunk_add_ons is provided but config is not, create config from list
        if chunk_add_ons and not chunk_addon_config:
            self.chunk_addon_config = ChunkAddOnConfig(enabled=chunk_add_ons)
        elif chunk_addon_config:
            # Merge any explicitly provided add-ons with config
            if chunk_add_ons:
                all_enabled = set(self.chunk_addon_config.enabled) | set(chunk_add_ons)
                self.chunk_addon_config.enabled = list(all_enabled)

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
            elif store_config.provider == "chroma":
                from spiderweb.stores.chroma import ChromaVectorStore

                persist_dir = None
                if hasattr(store_config, "extra_config") and store_config.extra_config:
                    persist_dir = store_config.extra_config.get("persist_directory")

                self.vector_store = ChromaVectorStore(
                    collection_name=store_config.collection_name,
                    persist_directory=persist_dir,
                    embedding_dimension=getattr(store_config, "embedding_dimension", 1536),
                )
                logger.info(f"Initialized Chroma vector store: {store_config.collection_name}")
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
            f"embedding={enable_embedding}, "
            f"add_ons={self.chunk_addon_config.enabled if self.chunk_addon_config.enabled else 'none'}, "
            f"graph_store={type(self.graph_store).__name__ if self.graph_store else 'none'}"
        )

    def _get_tiktoken_encoding(self):
        """Return tiktoken encoding for the current embedding model, or None if unavailable."""
        try:
            import tiktoken
        except ImportError:
            return None
        model = getattr(self.llm_client, "embedding_model", None) if self.llm_client else None
        if not isinstance(model, str):
            model = ""
        # OpenAI embedding models use cl100k_base; other providers may vary
        if "text-embedding-3" in model or "text-embedding-ada" in model or "embedding" in model.lower():
            try:
                return tiktoken.encoding_for_model("gpt-5.1")  # cl100k_base
            except Exception:
                return tiktoken.get_encoding("cl100k_base")
        return None

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Uses tiktoken when available and the embedding model is known (OpenAI-style);
        otherwise falls back to ~4 characters per token.

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        enc = self._get_tiktoken_encoding()
        if enc is not None:
            try:
                return len(enc.encode(text))
            except Exception:
                pass
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

    def _create_skipped_result_for_document(
        self,
        document: Document,
        start_time: float,
        reason: str,
    ) -> IngestionResult:
        """Create an IngestionResult for a skipped document (already loaded).

        Args:
            document: The document that was skipped
            start_time: Processing start time
            reason: Reason for skipping

        Returns:
            IngestionResult with success=True but no chunks
        """
        processing_time = time.time() - start_time
        return IngestionResult(
            document=document,
            success=True,
            chunks_created=0,
            chunks_validated=0,
            chunks_rejected=0,
            processing_time_seconds=processing_time,
            embedding_time_seconds=0.0,
            errors=[],
            warnings=[reason],
        )

    async def _run_pipeline_from_document(
        self,
        document: Document,
        store_chunks: bool,
        chunker_override: "Chunker | None",
        source_label: str,
    ) -> IngestionResult:
        """Run the pipeline from chunking through store on an already-loaded document.

        Used by process() after loading and by process_document() for crawl/ingest
        and search_crawl_extract ingest paths. Runs BEFORE_CHUNK through store.
        """
        start_time = time.time()
        errors: list[str] = []
        warnings: list[str] = []

        try:
            # Hook: BEFORE_CHUNK
            ctx = await self.hooks.run(
                HookPoint.BEFORE_CHUNK, document, file_path=source_label
            )
            if ctx.skip:
                logger.info(f"Chunking skipped by hook for {source_label}")
                return self._create_skipped_result_for_document(
                    document, start_time, "Skipped by BEFORE_CHUNK hook"
                )
            if ctx.modified_data is not None:
                document = ctx.modified_data

            # Chunk
            logger.debug("Step 2/5: Chunking document")
            chunker_to_use = chunker_override if chunker_override is not None else self.chunker
            if hasattr(chunker_to_use, "chunk_async"):
                chunks = await chunker_to_use.chunk_async(document)
            else:
                chunks = chunker_to_use.chunk(document)

            # Hook: AFTER_CHUNK
            ctx = await self.hooks.run(
                HookPoint.AFTER_CHUNK, chunks, document=document, file_path=source_label
            )
            if ctx.modified_data is not None:
                chunks = ctx.modified_data

            document.chunks = chunks
            chunks_created = len(chunks)
            logger.info(f"Created {chunks_created} chunks from {source_label}")

            # Run chunk add-ons
            if self.chunk_addon_config.enabled:
                logger.debug(
                    f"Step 2b/5: Running chunk add-ons: {self.chunk_addon_config.enabled}"
                )
                chunks = await self._run_chunk_add_ons(chunks, document)
                document.chunks = chunks

            # Propagate document source-scoping metadata to chunks (for vector query filter_dict)
            scope_attrs = _scope_attributes_from_document(document)
            if scope_attrs:
                for ch in chunks:
                    ch.metadata.extra = {**(ch.metadata.extra or {}), **scope_attrs}
                document.chunks = chunks

            # Graph store
            graph_entities_written: int | None = None
            graph_relationships_written: int | None = None
            if self.graph_store:
                entities, relationships = document_to_entities_and_relationships(
                    document
                )
                graph_entities_written = len(entities)
                graph_relationships_written = len(relationships)
                if entities:
                    await self.graph_store.upsert_entities(entities)
                if relationships:
                    await self.graph_store.upsert_relationships(relationships)
                if entities or relationships:
                    logger.info(
                        f"Graph store: upserted {len(entities)} entities, "
                        f"{len(relationships)} relationships"
                    )
                else:
                    logger.info(
                        "Graph store: no entities or relationships from document. "
                        "Ensure LangExtract add-on ran (pip install spiderweb[langextract])."
                    )

            # Validate
            chunks_validated = 0
            chunks_rejected = 0
            if self.enable_validation and self.validator:
                logger.debug(f"Step 3/5: Validating {len(chunks)} chunks")
                validation_results = await self.validator.validate_batch(chunks)
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

            # Embed
            embedding_start = time.time()
            if self.enable_embedding and self.llm_client:
                logger.debug(
                    f"Step 4/5: Generating embeddings for {len(chunks)} chunks"
                )
                chunks = await self._generate_embeddings(chunks)
                document.chunks = chunks
            else:
                logger.debug(
                    "Step 4/5: Skipping embedding generation (disabled or no LLM client)"
                )
            embedding_time = time.time() - embedding_start

            # Store
            if store_chunks and chunks:
                logger.debug(f"Step 5/5: Storing {len(chunks)} chunks in vector store")
                chunks_with_embeddings = [
                    c for c in chunks if c.embedding is not None
                ]
                if chunks_with_embeddings:
                    await self.vector_store.upsert(chunks_with_embeddings)
                    logger.info(
                        f"Stored {len(chunks_with_embeddings)} chunks in vector store"
                    )
                elif self.enable_embedding:
                    warnings.append("No chunks with embeddings to store")
            else:
                logger.debug("Step 5/5: Skipping vector store (disabled or no chunks)")

            processing_time = time.time() - start_time
            return IngestionResult(
                document=document,
                success=True,
                chunks_created=chunks_created,
                chunks_validated=chunks_validated,
                chunks_rejected=chunks_rejected,
                chunks_deduplicated=0,
                processing_time_seconds=processing_time,
                embedding_time_seconds=embedding_time,
                errors=errors,
                warnings=warnings,
                graph_entities_written=graph_entities_written,
                graph_relationships_written=graph_relationships_written,
            )

        except Exception as e:
            processing_time = time.time() - start_time
            err_detail = str(e).strip() or repr(e)
            error_msg = (
                f"Failed to process {source_label}: {type(e).__name__}: {err_detail}"
            )
            logger.error(error_msg, exc_info=True)
            errors.append(error_msg)
            try:
                from spiderweb.observability.metrics import (
                    increment_documents_processed,
                    record_ingest_duration,
                )
                record_ingest_duration(processing_time, success=False)
                increment_documents_processed(1, success=False)
            except Exception:
                pass
            return IngestionResult(
                document=document,
                success=False,
                chunks_created=0,
                chunks_validated=0,
                chunks_rejected=0,
                processing_time_seconds=processing_time,
                embedding_time_seconds=0.0,
                errors=errors,
                warnings=warnings,
            )

    async def process_document(
        self,
        document: Document,
        store_chunks: bool = True,
        chunker_override: "Chunker | None" = None,
    ) -> IngestionResult:
        """Process an already-loaded document through chunking, validation, embedding, and storage.

        Use this when the document was loaded elsewhere (e.g. WebLoader) so the
        pipeline runs from chunking onward. Hooks BEFORE_CHUNK and AFTER_CHUNK
        are run; BEFORE_EXTRACT and AFTER_EXTRACT are not.

        Args:
            document: Loaded document (e.g. from WebLoader.load).
            store_chunks: Whether to store chunks in the vector store.
            chunker_override: Optional chunker to use instead of self.chunker.

        Returns:
            Ingestion result with statistics.
        """
        source_label = document.metadata.source
        return await self._run_pipeline_from_document(
            document,
            store_chunks=store_chunks,
            chunker_override=chunker_override,
            source_label=source_label,
        )

    async def process(
        self,
        file_path: str | Path,
        store_chunks: bool = True,
        document_id: str | None = None,
        chunker_override: "Chunker | None" = None,
    ) -> IngestionResult:
        """Process a single document through the full pipeline.

        Args:
            file_path: Path to the document
            store_chunks: Whether to store chunks in vector store
            document_id: Optional custom document ID. If provided, this ID will be used
                for the document and all its chunks. If not provided, the auto-generated
                document ID from extraction will be used.
            chunker_override: Optional chunker to use instead of self.chunker.
                Useful for adaptive chunking where strategy is chosen per document.

        Returns:
            Ingestion result with statistics

        Raises:
            FileNotFoundError: If file doesn't exist
            ExtractionError: If extraction fails
        """
        path = Path(file_path)
        start_time = time.time()

        # Start tracing span
        from spiderweb.observability.tracing import span

        with span("process_document", {"file_path": str(path), "file_name": path.name}):
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

                # Override document ID if provided (allows external systems to control the ID)
                if document_id:
                    document.id = document_id
                    logger.debug(f"Using custom document ID: {document_id}")

                # Hook: AFTER_EXTRACT
                ctx = await self.hooks.run(HookPoint.AFTER_EXTRACT, document, file_path=str(path))
                if ctx.modified_data is not None:
                    document = ctx.modified_data

                return await self._run_pipeline_from_document(
                    document,
                    store_chunks=store_chunks,
                    chunker_override=chunker_override,
                    source_label=str(path),
                )

            except Exception as e:
                processing_time = time.time() - start_time
                err_detail = str(e).strip() or repr(e)
                error_msg = f"Failed to process {path.name}: {type(e).__name__}: {err_detail}"
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

                # Record metrics for failure
                try:
                    from spiderweb.observability.metrics import (
                        increment_documents_processed,
                        record_ingest_duration,
                    )

                    record_ingest_duration(processing_time, success=False)
                    increment_documents_processed(1, success=False)
                except Exception:
                    pass

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

    async def _run_chunk_add_ons(
        self,
        chunks: list[Chunk],
        document: Document,
    ) -> list[Chunk]:
        """Run enabled chunk add-ons on chunks.
        
        Args:
            chunks: List of chunks to process
            document: Parent document
            
        Returns:
            The same list of chunks (after mutation by add-ons)
        """
        if not self.chunk_addon_config.enabled:
            return chunks
        
        for addon_name in self.chunk_addon_config.enabled:
            # Get add-on options if configured
            addon_options = self.chunk_addon_config.options.get(addon_name, {})

            # Create add-on instance from registry
            if addon_name not in chunk_addon_registry:
                raise ValueError(
                    f"Unknown chunk add-on '{addon_name}'. "
                    f"Available: {chunk_addon_registry.list()}"
                )

            # Create add-on instance - pass llm_client and options
            addon_kwargs = {"llm_client": self.llm_client, **addon_options}
            addon = chunk_addon_registry.create(addon_name, **addon_kwargs)

            # Run add-on (prefer async, fall back to sync); on error log and continue with current chunks
            try:
                if hasattr(addon, "process_async"):
                    chunks = await addon.process_async(chunks, document=document)
                elif hasattr(addon, "process"):
                    chunks = addon.process(chunks, document=document)
                else:
                    raise ValueError(
                        f"Add-on '{addon_name}' does not implement process() or process_async()"
                    )
                logger.debug(f"Ran add-on '{addon_name}' on {len(chunks)} chunks")
            except Exception as e:
                logger.error("Chunk add-on '%s' failed: %s. Continuing with current chunks.", addon_name, e, exc_info=True)

        return chunks

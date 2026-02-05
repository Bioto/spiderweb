"""Batch processing for documents.

Provides efficient concurrent processing of multiple documents.
"""

import asyncio
import time
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gluellm import GlueLLM

from spiderweb.loaders.directory_loader import DirectoryLoader
from spiderweb.models.config import BatchConfig
from spiderweb.models.result import BatchIngestionResult, IngestionResult
from spiderweb.observability.logging_config import get_logger
from spiderweb.pipeline.processor import DocumentProcessor
from spiderweb.utils.ingest_cache import IngestCache

logger = get_logger(__name__)


class BatchProcessor:
    """Processor for batch document ingestion.

    Handles concurrent processing of multiple documents with progress tracking.

    Example:
        >>> from gluellm import GlueLLM
        >>> from spiderweb.pipeline import BatchProcessor
        >>>
        >>> llm = GlueLLM()
        >>> processor = BatchProcessor(llm_client=llm, max_concurrent=5)
        >>> result = await processor.process_directory("/path/to/docs")
        >>> print(f"Processed {result.successful_documents}/{result.total_documents}")
    """

    def __init__(
        self,
        llm_client: "GlueLLM | None" = None,
        document_processor: DocumentProcessor | None = None,
        config: BatchConfig | None = None,
    ):
        """Initialize batch processor.

        Args:
            llm_client: GlueLLM client for embeddings
            document_processor: Document processor instance
            config: Batch processing configuration
        """
        self.llm_client = llm_client
        self.config = config or BatchConfig()

        # Initialize document processor
        if document_processor:
            self.document_processor = document_processor
        else:
            self.document_processor = DocumentProcessor(llm_client=llm_client)

        # Initialize ingest cache if enabled
        self.cache: IngestCache | None = None
        if self.config.use_cache:
            self.cache = IngestCache()

        logger.info(
            f"Initialized BatchProcessor: "
            f"max_concurrent_extractions={self.config.max_concurrent_extractions}, "
            f"max_concurrent_embeddings={self.config.max_concurrent_embeddings}, "
            f"use_cache={self.config.use_cache}"
        )

    async def _process_single_document(
        self,
        file_path: Path,
        semaphore: asyncio.Semaphore,
    ) -> tuple[Path, IngestionResult]:
        """Process a single document with concurrency control.

        Args:
            file_path: Path to the document
            semaphore: Concurrency semaphore

        Returns:
            Tuple of (file_path, result)
        """
        async with semaphore:
            try:
                result = await self.document_processor.process(file_path)
                # Update cache on success
                if self.cache and result.success and result.document:
                    self.cache.update(file_path, result.document.id)
                return (file_path, result)
            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {e}", exc_info=True)

                # Return failed result with minimal document
                from spiderweb.models.document import Document, DocumentMetadata

                failed_doc = Document(
                    raw_content="",
                    markdown_content="",
                    metadata=DocumentMetadata(
                        source=str(file_path),
                        file_type="unknown",
                        extraction_method="none",
                    ),
                )

                return (
                    file_path,
                    IngestionResult(
                        document=failed_doc,
                        success=False,
                        chunks_created=0,
                        chunks_validated=0,
                        processing_time_seconds=0.0,
                        errors=[str(e)],
                    ),
                )

    async def process_files(
        self,
        file_paths: list[Path],
        show_progress: bool = True,
    ) -> BatchIngestionResult:
        """Process a list of files.

        Args:
            file_paths: List of file paths to process
            show_progress: Show progress logging

        Returns:
            Batch ingestion result
        """
        if not file_paths:
            logger.warning("No files to process")
            return BatchIngestionResult(
                total_documents=0,
                successful_documents=0,
                failed_documents=0,
                total_chunks=0,
                total_chunks_validated=0,
                processing_time_seconds=0.0,
                average_time_per_document=0.0,
                results=[],
            )

        # Filter files using cache if enabled
        files_to_process = file_paths
        if self.cache and not self.config.force:
            files_to_process = [
                path
                for path in file_paths
                if self.cache.should_process(path, force=False)
            ]
            skipped = len(file_paths) - len(files_to_process)
            if skipped > 0:
                logger.info(f"Skipping {skipped} unchanged files (use --force to re-process)")

        logger.info(f"Starting batch processing of {len(files_to_process)} files")

        start_time = time.time()
        started_at = datetime.now()

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.config.max_concurrent_extractions)

        # Process files concurrently
        tasks = [self._process_single_document(path, semaphore) for path in files_to_process]

        # Track progress
        results = []
        errors_by_file = {}

        for processed_count, coro in enumerate(asyncio.as_completed(tasks), 1):
            file_path, result = await coro
            results.append(result)

            if not result.success:
                errors_by_file[str(file_path)] = result.errors

            if show_progress and processed_count % 10 == 0:
                logger.info(f"Progress: {processed_count}/{len(files_to_process)} documents processed")

        # Save cache if enabled
        if self.cache:
            self.cache.save()

        # Calculate statistics
        processing_time = time.time() - start_time
        completed_at = datetime.now()

        successful_documents = sum(1 for r in results if r.success)
        failed_documents = len(results) - successful_documents
        total_chunks = sum(r.chunks_created for r in results if r.success)
        total_chunks_validated = sum(r.chunks_validated for r in results if r.success)
        total_chunks_rejected = sum(r.chunks_rejected for r in results if r.success)

        average_time = processing_time / len(results) if results else 0.0

        logger.info(
            f"Batch processing complete: {successful_documents}/{len(files_to_process)} successful, "
            f"{total_chunks} chunks created, {processing_time:.2f}s"
        )

        return BatchIngestionResult(
            total_documents=len(files_to_process),
            successful_documents=successful_documents,
            failed_documents=failed_documents,
            total_chunks=total_chunks,
            total_chunks_validated=total_chunks_validated,
            total_chunks_rejected=total_chunks_rejected,
            processing_time_seconds=processing_time,
            average_time_per_document=average_time,
            results=results,
            errors=errors_by_file,
            started_at=started_at,
            completed_at=completed_at,
        )

    async def process_directory(
        self,
        directory: str | Path,
        recursive: bool = True,
        show_progress: bool = True,
    ) -> BatchIngestionResult:
        """Process all documents in a directory.

        Args:
            directory: Path to the directory
            recursive: Process subdirectories recursively
            show_progress: Show progress logging

        Returns:
            Batch ingestion result

        Raises:
            FileNotFoundError: If directory doesn't exist
            ValueError: If path is not a directory
        """
        dir_path = Path(directory)

        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        if not dir_path.is_dir():
            raise ValueError(f"Path is not a directory: {dir_path}")

        logger.info(f"Loading files from directory: {dir_path} (recursive={recursive})")

        # Use directory loader to find files
        loader = DirectoryLoader(
            extractor=self.document_processor.extractor,
            max_concurrent=1,  # We'll handle concurrency in batch processor
            recursive=recursive,
            skip_on_error=self.config.continue_on_error,
        )

        # Get list of files
        files = loader._get_files(dir_path)

        if not files:
            logger.warning(f"No supported files found in {dir_path}")
            return BatchIngestionResult(
                total_documents=0,
                successful_documents=0,
                failed_documents=0,
                total_chunks=0,
                total_chunks_validated=0,
                processing_time_seconds=0.0,
                average_time_per_document=0.0,
                results=[],
            )

        logger.info(f"Found {len(files)} files to process")

        # Process files
        return await self.process_files(files, show_progress=show_progress)

    async def process_directory_streaming(
        self,
        directory: str | Path,
        recursive: bool = True,
    ) -> AsyncIterator[IngestionResult]:
        """Process directory with streaming results.

        Yields results as documents are processed.

        Args:
            directory: Path to the directory
            recursive: Process subdirectories recursively

        Yields:
            Ingestion results as they complete

        Raises:
            FileNotFoundError: If directory doesn't exist
            ValueError: If path is not a directory
        """
        dir_path = Path(directory)

        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        if not dir_path.is_dir():
            raise ValueError(f"Path is not a directory: {dir_path}")

        logger.info(f"Streaming processing from directory: {dir_path}")

        # Use directory loader
        loader = DirectoryLoader(
            extractor=self.document_processor.extractor,
            max_concurrent=1,
            recursive=recursive,
            skip_on_error=self.config.continue_on_error,
        )

        files = loader._get_files(dir_path)

        if not files:
            logger.warning(f"No supported files found in {dir_path}")
            return

        # Create semaphore for concurrency
        semaphore = asyncio.Semaphore(self.config.max_concurrent_extractions)

        # Process files and yield results as they complete
        tasks = [self._process_single_document(path, semaphore) for path in files]

        for coro in asyncio.as_completed(tasks):
            file_path, result = await coro
            yield result

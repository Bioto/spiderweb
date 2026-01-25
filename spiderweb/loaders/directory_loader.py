"""Directory loader for batch processing documents.

Provides functionality to load and process all documents in a directory.
"""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

from spiderweb.extractors.base import ExtractionError, Extractor
from spiderweb.extractors.markitdown import MarkitdownExtractor
from spiderweb.loaders.file_loader import FileLoader
from spiderweb.models.document import Document
from spiderweb.observability.logging_config import get_logger

logger = get_logger(__name__)


class DirectoryLoader:
    """Loader for processing all documents in a directory.

    Supports recursive directory traversal and concurrent processing.

    Example:
        >>> loader = DirectoryLoader(max_concurrent=5)
        >>> async for doc in loader.load_directory("/path/to/docs"):
        ...     print(f"Loaded {doc.metadata.source}")
    """

    def __init__(
        self,
        extractor: Extractor | None = None,
        max_concurrent: int = 5,
        recursive: bool = True,
        skip_on_error: bool = True,
    ):
        """Initialize directory loader.

        Args:
            extractor: Extractor to use (defaults to MarkitdownExtractor)
            max_concurrent: Maximum concurrent file processing
            recursive: Recursively process subdirectories
            skip_on_error: Continue processing if a file fails
        """
        self.extractor = extractor or MarkitdownExtractor()
        self.max_concurrent = max_concurrent
        self.recursive = recursive
        self.skip_on_error = skip_on_error
        self.file_loader = FileLoader(extractor=self.extractor)

        logger.debug(
            f"Initialized DirectoryLoader: max_concurrent={max_concurrent}, "
            f"recursive={recursive}, skip_on_error={skip_on_error}"
        )

    def _get_files(self, directory: Path) -> list[Path]:
        """Get all files in a directory.

        Args:
            directory: Directory to scan

        Returns:
            List of file paths
        """
        if self.recursive:
            # Recursively find all files
            files = [f for f in directory.rglob("*") if f.is_file()]
        else:
            # Only files in the top-level directory
            files = [f for f in directory.iterdir() if f.is_file()]

        # Filter to only supported files
        supported_files = [f for f in files if self.extractor.supports(f)]

        logger.info(f"Found {len(supported_files)} supported files in {directory} (total: {len(files)})")

        return supported_files

    async def _load_file(self, file_path: Path) -> Document | None:
        """Load a single file, handling errors.

        Args:
            file_path: Path to the file

        Returns:
            Document or None if loading failed and skip_on_error is True
        """
        try:
            return await self.file_loader.load(file_path)
        except (ExtractionError, FileNotFoundError) as e:
            if self.skip_on_error:
                logger.warning(f"Skipping {file_path.name}: {e}")
                return None
            raise

    async def load_directory(self, directory: str | Path) -> AsyncIterator[Document]:
        """Load all documents from a directory.

        Args:
            directory: Path to the directory

        Yields:
            Documents as they are processed

        Raises:
            FileNotFoundError: If directory doesn't exist
            ValueError: If path is not a directory
        """
        dir_path = Path(directory)

        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        if not dir_path.is_dir():
            raise ValueError(f"Path is not a directory: {dir_path}")

        logger.info(f"Loading directory: {dir_path}")

        # Get all files
        files = self._get_files(dir_path)

        if not files:
            logger.warning(f"No supported files found in {dir_path}")
            return

        # Process files with concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def process_file(file_path: Path) -> Document | None:
            async with semaphore:
                return await self._load_file(file_path)

        # Process files concurrently
        tasks = [process_file(f) for f in files]

        for coro in asyncio.as_completed(tasks):
            doc = await coro
            if doc is not None:
                yield doc

    async def load_all(self, directory: str | Path) -> list[Document]:
        """Load all documents from a directory into a list.

        Convenience method that collects all documents into memory.

        Args:
            directory: Path to the directory

        Returns:
            List of all processed documents

        Raises:
            FileNotFoundError: If directory doesn't exist
            ValueError: If path is not a directory
        """
        documents = []
        async for doc in self.load_directory(directory):
            documents.append(doc)

        logger.info(f"Loaded {len(documents)} documents from {directory}")
        return documents

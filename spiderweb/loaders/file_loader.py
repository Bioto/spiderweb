"""File loader for reading individual documents.

Provides functionality to load and extract content from single files.
"""

from pathlib import Path

from spiderweb.extractors.base import Extractor
from spiderweb.extractors.markitdown import MarkitdownExtractor
from spiderweb.models.document import Document
from spiderweb.observability.logging_config import get_logger

logger = get_logger(__name__)


class FileLoader:
    """Loader for processing individual files.

    Example:
        >>> loader = FileLoader()
        >>> doc = await loader.load("document.pdf")
        >>> print(f"Loaded {doc.metadata.file_type} with {len(doc.chunks)} chunks")
    """

    def __init__(self, extractor: Extractor | None = None):
        """Initialize file loader.

        Args:
            extractor: Extractor to use (defaults to MarkitdownExtractor)
        """
        self.extractor = extractor or MarkitdownExtractor()
        logger.debug(f"Initialized FileLoader with {type(self.extractor).__name__}")

    async def load(self, file_path: str | Path) -> Document:
        """Load and extract content from a file.

        Args:
            file_path: Path to the file to load

        Returns:
            Document with extracted content

        Raises:
            FileNotFoundError: If the file doesn't exist
            ExtractionError: If extraction fails
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        logger.info(f"Loading file: {path.name}")

        # Check if extractor supports this file
        if not self.extractor.supports(path.suffix):
            raise ValueError(f"Extractor {type(self.extractor).__name__} does not support {path.suffix}")

        # Extract content
        document = await self.extractor.extract(path)

        logger.debug(f"Loaded {path.name}: {len(document.raw_content)} characters")

        return document

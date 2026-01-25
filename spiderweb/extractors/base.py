"""Base extractor protocol and utilities.

Defines the interface that all extractors must implement.
"""

from pathlib import Path
from typing import Protocol, runtime_checkable

from spiderweb.models.document import Document


@runtime_checkable
class Extractor(Protocol):
    """Protocol for document extractors.

    All extractors must implement this interface to be compatible
    with the Spiderweb processing pipeline.
    """

    def supports(self, file_path: str | Path) -> bool:
        """Check if this extractor supports the given file.

        Args:
            file_path: Path to the file to check

        Returns:
            True if this extractor can process the file, False otherwise
        """
        ...

    async def extract(self, file_path: str | Path) -> Document:
        """Extract content from a document.

        Args:
            file_path: Path to the document to extract

        Returns:
            Document with extracted content and metadata

        Raises:
            FileNotFoundError: If the file doesn't exist
            ExtractionError: If extraction fails
        """
        ...


class ExtractionError(Exception):
    """Exception raised when document extraction fails."""

    pass


def get_file_type(file_path: str | Path) -> str:
    """Determine the file type from the file path.

    Args:
        file_path: Path to the file

    Returns:
        File extension without the dot (e.g., 'pdf', 'docx', 'md')
    """
    path = Path(file_path)
    return path.suffix.lstrip(".").lower()

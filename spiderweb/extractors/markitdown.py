"""Markitdown-based document extractor.

Uses the markitdown library to convert various document formats to markdown.
"""

import time
from pathlib import Path

from markitdown import MarkItDown

from spiderweb.extractors.base import ExtractionError, get_file_type
from spiderweb.models.document import Document, DocumentMetadata
from spiderweb.observability.logging_config import get_logger

logger = get_logger(__name__)


class MarkitdownExtractor:
    """Extractor that uses markitdown to convert documents to markdown.

    Supports a wide variety of document formats including PDF, DOCX, PPTX,
    XLSX, images (with OCR), HTML, and more.

    Example:
        >>> extractor = MarkitdownExtractor()
        >>> doc = await extractor.extract("document.pdf")
        >>> print(doc.markdown_content)
    """

    def __init__(self, enable_ocr: bool = False):
        """Initialize the markitdown extractor.

        Args:
            enable_ocr: Enable OCR for image processing (requires pytesseract)
        """
        self.enable_ocr = enable_ocr
        self._converter = MarkItDown()
        logger.debug(f"Initialized MarkitdownExtractor (OCR={'enabled' if enable_ocr else 'disabled'})")

    def supports(self, file_path: str | Path) -> bool:
        """Check if markitdown supports this file type.

        Markitdown supports: PDF, DOCX, PPTX, XLSX, images, HTML, and more.

        Args:
            file_path: Path to the file

        Returns:
            True if supported, False otherwise
        """
        supported_extensions = {
            "pdf",
            "docx",
            "doc",
            "pptx",
            "ppt",
            "xlsx",
            "xls",
            "html",
            "htm",
            "xml",
            "json",
            "csv",
            "md",
            "txt",
            "rtf",
            "jpg",
            "jpeg",
            "png",
            "gif",
            "bmp",
            "tiff",
            "wav",
            "mp3",
            "mp4",
        }

        file_type = get_file_type(file_path)
        return file_type in supported_extensions

    async def extract(self, file_path: str | Path) -> Document:
        """Extract content from a document using markitdown.

        Args:
            file_path: Path to the document

        Returns:
            Document with extracted markdown content

        Raises:
            FileNotFoundError: If the file doesn't exist
            ExtractionError: If extraction fails
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not self.supports(path):
            raise ExtractionError(f"Unsupported file type: {path.suffix}")

        start_time = time.time()
        logger.info(f"Extracting content from {path.name} using markitdown")

        try:
            # Convert document to markdown
            result = self._converter.convert(str(path))
            markdown_content = result.text_content

            # Get file stats
            file_stats = path.stat()
            file_size = file_stats.st_size

            # Create metadata
            metadata = DocumentMetadata(
                source=str(path.absolute()),
                file_type=get_file_type(path),
                extraction_method="markitdown",
                file_size_bytes=file_size,
            )

            elapsed_time = time.time() - start_time
            logger.info(
                f"Successfully extracted {len(markdown_content)} characters from {path.name} in {elapsed_time:.2f}s"
            )

            # Create document
            return Document(
                raw_content=markdown_content,
                markdown_content=markdown_content,
                metadata=metadata,
                extraction_results={"markitdown": markdown_content},
            )

        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"Extraction failed for {path.name} after {elapsed_time:.2f}s: {e}", exc_info=True)
            raise ExtractionError(f"Failed to extract content from {path}: {e}") from e

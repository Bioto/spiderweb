"""OCR-based PDF extractor using PyMuPDF and Tesseract.

For PDFs with broken text encoding or scanned documents.
"""

import io
from pathlib import Path

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

from spiderweb.models.document import Document, DocumentMetadata
from spiderweb.observability.logging_config import get_logger

logger = get_logger(__name__)


class OCRExtractor:
    """Extract text from PDFs using OCR.

    Useful for:
    - PDFs with broken text encoding (CID fonts)
    - Scanned documents
    - Image-based PDFs

    Example:
        >>> extractor = OCRExtractor(dpi=150)
        >>> document = await extractor.extract("document.pdf")
    """

    def __init__(
        self,
        dpi: int = 150,
        lang: str = "eng",
        max_pages: int | None = None,
    ):
        """Initialize OCR extractor.

        Args:
            dpi: Resolution for rendering PDF pages (higher = better quality, slower)
            lang: Tesseract language code (e.g., 'eng', 'fra', 'deu')
            max_pages: Maximum number of pages to process (None = all pages)
        """
        self.dpi = dpi
        self.lang = lang
        self.max_pages = max_pages

        logger.debug(f"Initialized OCRExtractor: dpi={dpi}, lang={lang}")

    async def extract(self, source: str | Path) -> Document:
        """Extract text from PDF using OCR.

        Args:
            source: Path to PDF file

        Returns:
            Document with extracted text

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not a PDF
        """
        path = Path(source)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        file_suffix = path.suffix.lower()
        if file_suffix != ".pdf":
            raise ValueError(f"OCRExtractor only supports PDF files, got: {file_suffix}")

        logger.info(f"Starting OCR extraction for {path.name}")

        # Open PDF
        doc = fitz.open(path)
        total_pages = len(doc)
        pages_to_process = min(total_pages, self.max_pages) if self.max_pages else total_pages

        logger.info(f"Processing {pages_to_process}/{total_pages} pages")

        # Extract text from each page
        extracted_pages = []

        for page_num in range(pages_to_process):
            logger.debug(f"OCR processing page {page_num + 1}/{pages_to_process}")

            page = doc[page_num]

            # Render page as image
            pix = page.get_pixmap(dpi=self.dpi)
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))

            # OCR the image
            page_text = pytesseract.image_to_string(img, lang=self.lang)
            extracted_pages.append(page_text)

        doc.close()

        # Combine all pages
        raw_content = "\n\n".join(extracted_pages)

        # For OCR, raw and markdown are the same
        markdown_content = raw_content

        logger.info(f"OCR extraction complete: {len(raw_content)} characters from {pages_to_process} pages")

        # Create metadata
        metadata = DocumentMetadata(
            source=str(path),
            file_type="pdf",
            extraction_method="ocr",
            extra={
                "pages_processed": pages_to_process,
                "total_pages": total_pages,
                "dpi": self.dpi,
                "lang": self.lang,
            },
        )

        # Create document
        return Document(
            raw_content=raw_content,
            markdown_content=markdown_content,
            metadata=metadata,
        )

    def supports(self, file_type: str) -> bool:
        """Check if this extractor supports the file type.

        Args:
            file_type: File extension (with or without dot)

        Returns:
            True if file type is supported
        """
        # Handle both string and Path objects, and with/without dot
        ft = str(file_type).lower().lstrip(".")
        return ft == "pdf"

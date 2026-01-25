"""Document extractors for Spiderweb.

This package provides extractors for converting various document formats
into text that can be chunked and embedded.
"""

from spiderweb.extractors.base import Extractor
from spiderweb.extractors.markitdown import MarkitdownExtractor
from spiderweb.extractors.ocr import OCRExtractor

__all__ = ["Extractor", "MarkitdownExtractor", "OCRExtractor"]

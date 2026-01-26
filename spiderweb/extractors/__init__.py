"""Document extractors for Spiderweb.

This package provides extractors for converting various document formats
into text that can be chunked and embedded.
"""

from spiderweb.extractors.base import Extractor
from spiderweb.extractors.markitdown import MarkitdownExtractor

# OCR extractor is imported lazily to avoid requiring optional dependencies
try:
    from spiderweb.extractors.ocr import OCRExtractor
    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False
    OCRExtractor = None  # type: ignore

__all__ = ["Extractor", "MarkitdownExtractor"]

if _OCR_AVAILABLE:
    __all__.append("OCRExtractor")

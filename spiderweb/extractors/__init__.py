"""Document extractors for Spiderweb.

This package provides extractors for converting various document formats
into text that can be chunked and embedded.

Built-in extractors are automatically registered in the global extractor_registry.
Custom extractors can be registered for use via configuration strings.
"""

from spiderweb.extractors.base import Extractor
from spiderweb.extractors.markitdown import MarkitdownExtractor
from spiderweb.registry import extractor_registry

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

# Register built-in extractors
extractor_registry.register("markitdown", MarkitdownExtractor)

if _OCR_AVAILABLE and OCRExtractor is not None:
    extractor_registry.register("ocr", OCRExtractor)

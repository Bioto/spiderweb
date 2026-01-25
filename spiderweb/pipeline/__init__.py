"""Processing pipeline for Spiderweb.

This package provides the main orchestration for document processing,
from extraction through chunking, validation, and storage.
"""

from spiderweb.pipeline.batch import BatchProcessor
from spiderweb.pipeline.processor import DocumentProcessor

__all__ = ["DocumentProcessor", "BatchProcessor"]

"""Chunking strategies for Spiderweb.

This package provides various chunking strategies for splitting documents
into manageable pieces for embedding and retrieval.
"""

from spiderweb.chunkers.base import Chunker
from spiderweb.chunkers.hierarchical import HierarchicalChunker
from spiderweb.chunkers.semantic import SemanticChunker
from spiderweb.chunkers.sentence import SentenceChunker
from spiderweb.chunkers.sliding_window import SlidingWindowChunker

__all__ = [
    "Chunker",
    "SlidingWindowChunker",
    "SemanticChunker",
    "HierarchicalChunker",
    "SentenceChunker",
]

"""Chunking strategies for Spiderweb.

This package provides various chunking strategies for splitting documents
into manageable pieces for embedding and retrieval.

Built-in chunkers are automatically registered in the global chunker_registry.
Custom chunkers can be registered for use via configuration strings.
"""

from spiderweb.chunkers.base import Chunker
from spiderweb.chunkers.hierarchical import HierarchicalChunker
from spiderweb.chunkers.semantic import SemanticChunker
from spiderweb.chunkers.sentence import SentenceChunker
from spiderweb.chunkers.sliding_window import SlidingWindowChunker
from spiderweb.registry import chunker_registry

__all__ = [
    "Chunker",
    "SlidingWindowChunker",
    "SemanticChunker",
    "HierarchicalChunker",
    "SentenceChunker",
]

# Register built-in chunkers
# These names correspond to ChunkType enum values
chunker_registry.register("hierarchical", HierarchicalChunker)
chunker_registry.register("sentence", SentenceChunker)
chunker_registry.register("semantic", SemanticChunker)
chunker_registry.register("sliding_window", SlidingWindowChunker)

"""Vector stores for Spiderweb.

This package provides vector storage implementations for persisting
and querying document chunks with embeddings.
"""

from spiderweb.stores.base import VectorStore
from spiderweb.stores.memory import MemoryVectorStore
from spiderweb.stores.qdrant import QdrantVectorStore

__all__ = ["VectorStore", "MemoryVectorStore", "QdrantVectorStore"]

# Chroma is optional (requires chromadb)
try:
    from spiderweb.stores.chroma import ChromaVectorStore

    __all__.append("ChromaVectorStore")
except ImportError:
    # chromadb not installed
    pass

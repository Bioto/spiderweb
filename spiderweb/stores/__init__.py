"""Vector stores for Spiderweb.

This package provides vector storage implementations for persisting
and querying document chunks with embeddings.
"""

from spiderweb.stores.base import VectorStore
from spiderweb.stores.memory import MemoryVectorStore
from spiderweb.stores.qdrant import QdrantVectorStore

__all__ = ["VectorStore", "MemoryVectorStore", "QdrantVectorStore"]

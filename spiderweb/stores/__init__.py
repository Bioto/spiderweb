"""Vector and graph stores for Spiderweb.

Vector stores persist document chunks with embeddings; graph stores
persist entities and relationships from add-ons (e.g. Neo4j).
"""

from spiderweb.stores.base import VectorStore
from spiderweb.stores.graph_base import GraphStore
from spiderweb.stores.memory import MemoryVectorStore
from spiderweb.stores.qdrant import QdrantVectorStore

__all__ = ["VectorStore", "GraphStore", "MemoryVectorStore", "QdrantVectorStore"]

# Chroma is optional (requires chromadb)
try:
    from spiderweb.stores.chroma import ChromaVectorStore

    __all__.append("ChromaVectorStore")
except ImportError:
    pass

# Neo4j graph store is optional (requires neo4j driver)
try:
    from spiderweb.stores.neo4j import Neo4jGraphStore

    __all__.append("Neo4jGraphStore")
except ImportError:
    pass

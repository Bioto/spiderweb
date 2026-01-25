"""In-memory vector store for testing and development.

Simple implementation that stores chunks in memory without persistence.
"""

from spiderweb.models.document import Chunk
from spiderweb.observability.logging_config import get_logger

logger = get_logger(__name__)


class MemoryVectorStore:
    """In-memory vector store.

    Simple implementation for testing and development. Does not persist data.

    Example:
        >>> store = MemoryVectorStore()
        >>> await store.upsert(chunks)
        >>> results = await store.query(query_embedding, top_k=5)
    """

    def __init__(self):
        """Initialize in-memory store."""
        self._chunks: dict[str, Chunk] = {}
        logger.debug("Initialized MemoryVectorStore")

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity (-1 to 1)
        """
        import math

        dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=True))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    async def upsert(self, chunks: list[Chunk]) -> None:
        """Insert or update chunks.

        Args:
            chunks: Chunks to upsert

        Raises:
            ValueError: If chunks don't have embeddings
        """
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk {chunk.id} does not have an embedding")

            self._chunks[chunk.id] = chunk

        logger.debug(f"Upserted {len(chunks)} chunks (total: {len(self._chunks)})")

    async def query(
        self,
        embedding: list[float],
        top_k: int = 10,
        filter_dict: dict | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Query for similar chunks.

        Args:
            embedding: Query embedding
            top_k: Number of results
            filter_dict: Metadata filters (not implemented for memory store)

        Returns:
            List of (chunk, score) tuples
        """
        if not self._chunks:
            return []

        # Calculate similarities
        similarities = []
        for chunk in self._chunks.values():
            if chunk.embedding:
                similarity = self._cosine_similarity(embedding, chunk.embedding)
                similarities.append((chunk, similarity))

        # Sort by similarity (descending) and take top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = similarities[:top_k]

        logger.debug(f"Query returned {len(results)} results")

        return results

    async def delete(self, chunk_ids: list[str]) -> None:
        """Delete chunks.

        Args:
            chunk_ids: IDs to delete
        """
        deleted = 0
        for chunk_id in chunk_ids:
            if chunk_id in self._chunks:
                del self._chunks[chunk_id]
                deleted += 1

        logger.debug(f"Deleted {deleted} chunks")

    async def get(self, chunk_ids: list[str]) -> list[Chunk]:
        """Get chunks by ID.

        Args:
            chunk_ids: IDs to retrieve

        Returns:
            List of chunks
        """
        chunks = []
        for chunk_id in chunk_ids:
            if chunk_id in self._chunks:
                chunks.append(self._chunks[chunk_id])

        logger.debug(f"Retrieved {len(chunks)}/{len(chunk_ids)} chunks")

        return chunks

    async def count(self) -> int:
        """Get total chunk count.

        Returns:
            Number of chunks
        """
        return len(self._chunks)

    async def clear(self) -> None:
        """Clear all chunks."""
        count = len(self._chunks)
        self._chunks.clear()
        logger.info(f"Cleared {count} chunks from memory store")

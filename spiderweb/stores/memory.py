"""In-memory vector store for testing and development.

Simple implementation that stores chunks in memory without persistence.
"""

from typing import Literal

from spiderweb.models.document import Chunk
from spiderweb.observability.logging_config import get_logger
from spiderweb.utils.vector_math import cosine_similarity

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
                similarity = cosine_similarity(embedding, chunk.embedding)
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

    async def get_by_position(
        self,
        document_id: str,
        position_start: int,
        position_end: int,
        position_field: Literal["chunk_index", "page_number"] = "chunk_index",
    ) -> list[Chunk]:
        """Retrieve chunks by position range within a document.

        Args:
            document_id: Document to search within
            position_start: Starting position (inclusive)
            position_end: Ending position (inclusive)
            position_field: Whether to use chunk_index or page_number

        Returns:
            List of chunks in the position range, sorted by position
        """
        result_chunks = []

        for chunk in self._chunks.values():
            # Filter by document_id
            if chunk.metadata.document_id != document_id:
                continue

            # Filter by position range
            if position_field == "chunk_index":
                pos = chunk.metadata.chunk_index
                if position_start <= pos <= position_end:
                    result_chunks.append(chunk)
            elif position_field == "page_number":
                # Check if any page_number in range
                if chunk.metadata.page_numbers:
                    if any(position_start <= page <= position_end for page in chunk.metadata.page_numbers):
                        result_chunks.append(chunk)

        # Sort by position
        if position_field == "chunk_index":
            result_chunks.sort(key=lambda c: c.metadata.chunk_index)
        elif position_field == "page_number":
            # Sort by first page number
            result_chunks.sort(key=lambda c: c.metadata.page_numbers[0] if c.metadata.page_numbers else 0)

        logger.debug(
            f"Retrieved {len(result_chunks)} chunks by {position_field} "
            f"range [{position_start}, {position_end}] for document {document_id}"
        )

        return result_chunks

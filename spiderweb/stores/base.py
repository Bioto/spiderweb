"""Base vector store protocol.

Defines the interface for vector storage implementations.
"""

from typing import Literal, Protocol, runtime_checkable

from spiderweb.models.document import Chunk


@runtime_checkable
class VectorStore(Protocol):
    """Protocol for vector storage implementations.

    All vector stores must implement this interface to be compatible
    with the Spiderweb processing pipeline.
    """

    async def upsert(self, chunks: list[Chunk]) -> None:
        """Insert or update chunks in the vector store.

        Args:
            chunks: List of chunks to upsert (must have embeddings)

        Raises:
            ValueError: If chunks don't have embeddings
        """
        ...

    async def query(
        self,
        embedding: list[float],
        top_k: int = 10,
        filter_dict: dict | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Query the vector store for similar chunks.

        Args:
            embedding: Query embedding vector
            top_k: Number of results to return
            filter_dict: Optional metadata filters

        Returns:
            List of (chunk, similarity_score) tuples, sorted by similarity
        """
        ...

    async def delete(self, chunk_ids: list[str]) -> None:
        """Delete chunks from the vector store.

        Args:
            chunk_ids: IDs of chunks to delete
        """
        ...

    async def get(self, chunk_ids: list[str]) -> list[Chunk]:
        """Retrieve chunks by ID.

        Args:
            chunk_ids: IDs of chunks to retrieve

        Returns:
            List of chunks (may be shorter if some IDs not found)
        """
        ...

    async def get_by_position(
        self,
        document_id: str,
        position_start: int,
        position_end: int,
        position_field: Literal["chunk_index", "page_number"] = "chunk_index",
    ) -> list[Chunk]:
        """Retrieve chunks by position range within a document.
        
        Used for context window retrieval to get surrounding chunks.

        Args:
            document_id: Document to search within
            position_start: Starting position (inclusive)
            position_end: Ending position (inclusive)
            position_field: Whether to use chunk_index or page_number

        Returns:
            List of chunks in the position range, sorted by position
        """
        ...

    async def count(self) -> int:
        """Get the total number of chunks in the store.

        Returns:
            Number of chunks
        """
        ...

    async def clear(self) -> None:
        """Delete all chunks from the store.

        Use with caution!
        """
        ...

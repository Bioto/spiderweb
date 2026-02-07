"""Base protocol and utilities for chunk add-ons.

Defines the interface that all chunk add-ons must implement.
"""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from spiderweb.models.document import Chunk, Document


@runtime_checkable
class ChunkAddOn(Protocol):
    """Protocol for chunk add-ons.

    Chunk add-ons enrich chunks with additional data after chunking.
    They can attach data to chunks via chunk.metadata.extra.

    Add-ons can be synchronous or asynchronous. The pipeline will
    handle both appropriately.

    Example:
        >>> class MyAddOn:
        ...     async def process(self, chunks: list[Chunk], *, document: Document | None = None, **kwargs) -> list[Chunk]:
        ...         for chunk in chunks:
        ...             chunk.metadata.extra["my_key"] = "my_value"
        ...         return chunks
    """

    def process(
        self,
        chunks: "list[Chunk]",
        *,
        document: "Document | None" = None,
        **kwargs: object,
    ) -> "list[Chunk]":
        """Process chunks synchronously.

        Mutates chunks in place by adding data to chunk.metadata.extra,
        then returns the same list.

        Args:
            chunks: List of chunks to process
            document: Optional parent document (for context)
            **kwargs: Additional context from the pipeline

        Returns:
            The same list of chunks (after mutation)
        """
        ...

    async def process_async(
        self,
        chunks: "list[Chunk]",
        *,
        document: "Document | None" = None,
        **kwargs: object,
    ) -> "list[Chunk]":
        """Process chunks asynchronously.

        Mutates chunks in place by adding data to chunk.metadata.extra,
        then returns the same list.

        Args:
            chunks: List of chunks to process
            document: Optional parent document (for context)
            **kwargs: Additional context from the pipeline

        Returns:
            The same list of chunks (after mutation)
        """
        ...

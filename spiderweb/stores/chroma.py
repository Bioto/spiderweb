"""Chroma vector store implementation.

Provides persistent vector storage using ChromaDB.
"""

from typing import Any, Literal

from spiderweb.models.document import Chunk, ChunkMetadata
from spiderweb.observability.logging_config import get_logger

logger = get_logger(__name__)


class ChromaVectorStore:
    """ChromaDB-based vector store.

    Provides persistent storage and efficient similarity search using ChromaDB.

    Example:
        >>> store = ChromaVectorStore(
        ...     collection_name="my_docs",
        ...     persist_directory="./chroma_db",
        ... )
        >>> await store.initialize()
        >>> await store.upsert(chunks)
        >>> results = await store.query(query_embedding, top_k=5)
    """

    def __init__(
        self,
        collection_name: str = "spiderweb_documents",
        persist_directory: str | None = None,
        embedding_dimension: int = 1536,
        **kwargs: Any,
    ):
        """Initialize Chroma vector store.

        Args:
            collection_name: Name of the collection to use
            persist_directory: Directory to persist data (None = in-memory)
            embedding_dimension: Dimension of embedding vectors
            **kwargs: Additional Chroma client options
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_dimension = embedding_dimension
        self._kwargs = kwargs

        self._client = None
        self._collection = None
        self._initialized = False

        logger.debug(
            f"Initialized ChromaVectorStore: collection={collection_name}, "
            f"persist_directory={persist_directory}, dimension={embedding_dimension}"
        )

    async def initialize(self) -> None:
        """Initialize Chroma client and collection."""
        if self._initialized:
            return

        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as e:
            raise ImportError(
                "chromadb is not installed. Install it with: pip install chromadb"
            ) from e

        # Create client
        if self.persist_directory:
            self._client = chromadb.PersistentClient(
                path=self.persist_directory, **self._kwargs
            )
        else:
            self._client = chromadb.Client(Settings(**self._kwargs))

        # Get or create collection
        try:
            self._collection = self._client.get_collection(name=self.collection_name)
            logger.debug(f"Retrieved existing collection: {self.collection_name}")
        except Exception:
            # Collection doesn't exist, create it
            self._collection = self._client.create_collection(
                name=self.collection_name,
                metadata={"embedding_dimension": self.embedding_dimension},
            )
            logger.debug(f"Created new collection: {self.collection_name}")

        self._initialized = True

    async def upsert(self, chunks: list[Chunk]) -> None:
        """Insert or update chunks.

        Args:
            chunks: Chunks to upsert (must have embeddings)

        Raises:
            ValueError: If chunks don't have embeddings
            RuntimeError: If store not initialized
        """
        if not self._initialized:
            await self.initialize()

        if not chunks:
            return

        # Validate embeddings
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk {chunk.id} does not have an embedding")

        # Prepare data for Chroma
        ids = [chunk.id for chunk in chunks]
        embeddings = [chunk.embedding for chunk in chunks if chunk.embedding]
        documents = [chunk.content for chunk in chunks]
        metadatas = []

        for chunk in chunks:
            metadata: dict[str, Any] = {
                "document_id": chunk.metadata.document_id,
                "chunk_index": chunk.metadata.chunk_index,
                "chunk_type": chunk.metadata.chunk_type.value,
            }
            # Add optional metadata fields
            if chunk.metadata.start_char is not None:
                metadata["start_char"] = chunk.metadata.start_char
            if chunk.metadata.end_char is not None:
                metadata["end_char"] = chunk.metadata.end_char
            if chunk.metadata.page_numbers:
                metadata["page_numbers"] = str(chunk.metadata.page_numbers)
            if chunk.metadata.section_title:
                metadata["section_title"] = chunk.metadata.section_title
            if chunk.metadata.section_level is not None:
                metadata["section_level"] = chunk.metadata.section_level
            metadatas.append(metadata)

        # Upsert to Chroma
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        logger.debug(f"Upserted {len(chunks)} chunks to Chroma collection {self.collection_name}")

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
            filter_dict: Metadata filters (Chroma where clause format)

        Returns:
            List of (chunk, score) tuples

        Raises:
            RuntimeError: If store not initialized
        """
        if not self._initialized:
            await self.initialize()

        # Build where clause from filter_dict
        where = None
        if filter_dict:
            # Convert filter_dict to Chroma where clause
            # Chroma uses {"key": {"$eq": value}} format
            where = {}
            for key, value in filter_dict.items():
                if isinstance(value, list):
                    where[key] = {"$in": value}
                else:
                    where[key] = {"$eq": value}

        # Query Chroma
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=where,
        )

        # Convert results to Chunk objects
        chunks_with_scores = []
        if results["ids"] and len(results["ids"][0]) > 0:
            for i, chunk_id in enumerate(results["ids"][0]):
                metadata_dict = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0.0
                # Chroma returns distance (lower is better), convert to similarity
                similarity = 1.0 - distance

                # Reconstruct ChunkMetadata
                chunk_metadata = ChunkMetadata(
                    document_id=metadata_dict.get("document_id", ""),
                    chunk_index=metadata_dict.get("chunk_index", 0),
                    chunk_type=metadata_dict.get("chunk_type", "sliding_window"),
                    start_char=metadata_dict.get("start_char"),
                    end_char=metadata_dict.get("end_char"),
                    section_title=metadata_dict.get("section_title"),
                    section_level=metadata_dict.get("section_level"),
                )

                # Reconstruct Chunk
                chunk = Chunk(
                    id=chunk_id,
                    content=results["documents"][0][i] if results["documents"] else "",
                    metadata=chunk_metadata,
                    embedding=results["embeddings"][0][i] if results["embeddings"] else None,
                )

                chunks_with_scores.append((chunk, similarity))

        logger.debug(f"Chroma query returned {len(chunks_with_scores)} results")
        return chunks_with_scores

    async def delete(self, chunk_ids: list[str]) -> None:
        """Delete chunks.

        Args:
            chunk_ids: IDs to delete

        Raises:
            RuntimeError: If store not initialized
        """
        if not self._initialized:
            await self.initialize()

        if chunk_ids:
            self._collection.delete(ids=chunk_ids)
            logger.debug(f"Deleted {len(chunk_ids)} chunks from Chroma")

    async def delete_by_document_id(self, document_id: str) -> int:
        """Delete all chunks belonging to a document.

        Args:
            document_id: The document ID whose chunks should be deleted

        Returns:
            Number of chunks deleted

        Raises:
            RuntimeError: If store not initialized
        """
        if not self._initialized:
            await self.initialize()

        # Query for all chunks with this document_id
        results = self._collection.get(
            where={"document_id": {"$eq": document_id}},
        )

        if results["ids"]:
            self._collection.delete(ids=results["ids"])
            logger.debug(f"Deleted {len(results['ids'])} chunks for document {document_id}")
            return len(results["ids"])

        return 0

    async def get(self, chunk_ids: list[str]) -> list[Chunk]:
        """Get chunks by ID.

        Args:
            chunk_ids: IDs to retrieve

        Returns:
            List of chunks

        Raises:
            RuntimeError: If store not initialized
        """
        if not self._initialized:
            await self.initialize()

        if not chunk_ids:
            return []

        results = self._collection.get(ids=chunk_ids)

        chunks = []
        if results["ids"]:
            for i, chunk_id in enumerate(results["ids"]):
                metadata_dict = results["metadatas"][i] if results["metadatas"] else {}
                chunk_metadata = ChunkMetadata(
                    document_id=metadata_dict.get("document_id", ""),
                    chunk_index=metadata_dict.get("chunk_index", 0),
                    chunk_type=metadata_dict.get("chunk_type", "sliding_window"),
                )

                chunk = Chunk(
                    id=chunk_id,
                    content=results["documents"][i] if results["documents"] else "",
                    metadata=chunk_metadata,
                    embedding=results["embeddings"][i] if results["embeddings"] else None,
                )
                chunks.append(chunk)

        return chunks

    async def count(self) -> int:
        """Get total chunk count.

        Returns:
            Number of chunks

        Raises:
            RuntimeError: If store not initialized
        """
        if not self._initialized:
            await self.initialize()

        return self._collection.count()

    async def clear(self) -> None:
        """Clear all chunks.

        Raises:
            RuntimeError: If store not initialized
        """
        if not self._initialized:
            await self.initialize()

        # Delete collection and recreate
        self._client.delete_collection(name=self.collection_name)
        self._collection = self._client.create_collection(
            name=self.collection_name,
            metadata={"embedding_dimension": self.embedding_dimension},
        )
        logger.info(f"Cleared Chroma collection {self.collection_name}")

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

        Raises:
            RuntimeError: If store not initialized
        """
        if not self._initialized:
            await self.initialize()

        # Query for chunks with this document_id
        where = {"document_id": {"$eq": document_id}}
        results = self._collection.get(where=where)

        if not results["ids"]:
            return []

        # Filter by position range
        chunks = []
        for i, chunk_id in enumerate(results["ids"]):
            metadata_dict = results["metadatas"][i] if results["metadatas"] else {}

            if position_field == "chunk_index":
                chunk_index = metadata_dict.get("chunk_index", 0)
                if position_start <= chunk_index <= position_end:
                    chunk_metadata = ChunkMetadata(
                        document_id=metadata_dict.get("document_id", ""),
                        chunk_index=chunk_index,
                        chunk_type=metadata_dict.get("chunk_type", "sliding_window"),
                    )
                    chunk = Chunk(
                        id=chunk_id,
                        content=results["documents"][i] if results["documents"] else "",
                        metadata=chunk_metadata,
                        embedding=results["embeddings"][i] if results["embeddings"] else None,
                    )
                    chunks.append(chunk)
            elif position_field == "page_number":
                page_numbers_str = metadata_dict.get("page_numbers")
                if page_numbers_str:
                    # Parse page_numbers (stored as string)
                    import ast

                    page_numbers = ast.literal_eval(page_numbers_str)
                    if any(position_start <= page <= position_end for page in page_numbers):
                        chunk_metadata = ChunkMetadata(
                            document_id=metadata_dict.get("document_id", ""),
                            chunk_index=metadata_dict.get("chunk_index", 0),
                            chunk_type=metadata_dict.get("chunk_type", "sliding_window"),
                            page_numbers=page_numbers,
                        )
                        chunk = Chunk(
                            id=chunk_id,
                            content=results["documents"][i] if results["documents"] else "",
                            metadata=chunk_metadata,
                            embedding=results["embeddings"][i] if results["embeddings"] else None,
                        )
                        chunks.append(chunk)

        # Sort by position
        if position_field == "chunk_index":
            chunks.sort(key=lambda c: c.metadata.chunk_index)
        elif position_field == "page_number":
            chunks.sort(
                key=lambda c: c.metadata.page_numbers[0] if c.metadata.page_numbers else 0
            )

        return chunks

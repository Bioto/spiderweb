"""Qdrant vector store implementation.

Provides persistent vector storage using Qdrant.
"""

from typing import Any, Literal

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels

from spiderweb.models.config import VectorStoreConfig
from spiderweb.models.document import Chunk, ChunkMetadata, ChunkType
from spiderweb.observability.logging_config import get_logger

logger = get_logger(__name__)


class QdrantVectorStore:
    """Qdrant-based vector store.

    Provides persistent storage and efficient similarity search using Qdrant.

    Example:
        >>> store = QdrantVectorStore(
        ...     host="localhost",
        ...     port=6333,
        ...     collection_name="my_docs",
        ...     embedding_dimension=1536,
        ... )
        >>> await store.initialize()
        >>> await store.upsert(chunks)
        >>> results = await store.query(query_embedding, top_k=5)
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "spiderweb_documents",
        embedding_dimension: int = 1536,
        distance_metric: str = "cosine",
        api_key: str | None = None,
        use_https: bool = False,
    ):
        """Initialize Qdrant vector store.

        Args:
            host: Qdrant server host
            port: Qdrant server port
            collection_name: Name of the collection to use
            embedding_dimension: Dimension of embedding vectors
            distance_metric: Distance metric ('cosine', 'euclidean', 'dot')
            api_key: API key for cloud deployments
            use_https: Use HTTPS for connection
        """
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        self.distance_metric = distance_metric

        # Create client
        self.client = AsyncQdrantClient(
            host=host,
            port=port,
            api_key=api_key,
            https=use_https,
        )

        self._initialized = False

        logger.debug(
            f"Initialized QdrantVectorStore: collection={collection_name}, "
            f"host={host}:{port}, dimension={embedding_dimension}"
        )

    @classmethod
    def from_config(cls, config: VectorStoreConfig) -> "QdrantVectorStore":
        """Create store from configuration.

        Args:
            config: Vector store configuration

        Returns:
            Configured store instance
        """
        return cls(
            host=config.host,
            port=config.port,
            collection_name=config.collection_name,
            embedding_dimension=config.embedding_dimension,
            distance_metric=config.distance_metric,
            api_key=config.api_key,
            use_https=config.use_https,
        )

    def _get_distance_metric(self) -> qmodels.Distance:
        """Get Qdrant distance metric enum.

        Returns:
            Qdrant distance metric
        """
        metric_map = {
            "cosine": qmodels.Distance.COSINE,
            "euclidean": qmodels.Distance.EUCLID,
            "dot": qmodels.Distance.DOT,
        }
        return metric_map.get(self.distance_metric.lower(), qmodels.Distance.COSINE)

    async def initialize(self) -> None:
        """Initialize the collection.

        Creates the collection if it doesn't exist.
        """
        if self._initialized:
            return

        # Check if collection exists
        collections = await self.client.get_collections()
        collection_names = [c.name for c in collections.collections]

        if self.collection_name not in collection_names:
            # Create collection
            logger.info(f"Creating Qdrant collection: {self.collection_name}")

            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self.embedding_dimension,
                    distance=self._get_distance_metric(),
                ),
            )

            logger.info(f"Created collection {self.collection_name}")
        else:
            logger.debug(f"Collection {self.collection_name} already exists")

        self._initialized = True

    def _chunk_to_payload(self, chunk: Chunk) -> dict[str, Any]:
        """Convert chunk to Qdrant payload.

        Args:
            chunk: Chunk to convert

        Returns:
            Payload dictionary
        """
        # Serialize chunk metadata
        payload = {
            "chunk_id": chunk.id,
            "content": chunk.content,
            "document_id": chunk.metadata.document_id,
            "chunk_index": chunk.metadata.chunk_index,
            "chunk_type": chunk.metadata.chunk_type.value,
            "validation_scores": chunk.validation_scores,
            "parent_id": chunk.parent_id,
            "children_ids": chunk.children_ids,
        }

        # Add optional metadata fields
        if chunk.metadata.start_char is not None:
            payload["start_char"] = chunk.metadata.start_char
        if chunk.metadata.end_char is not None:
            payload["end_char"] = chunk.metadata.end_char
        if chunk.metadata.page_numbers:
            payload["page_numbers"] = chunk.metadata.page_numbers
        if chunk.metadata.section_title:
            payload["section_title"] = chunk.metadata.section_title
        if chunk.metadata.section_level is not None:
            payload["section_level"] = chunk.metadata.section_level
        if chunk.metadata.extra:
            payload["extra"] = chunk.metadata.extra

        return payload

    def _payload_to_chunk(self, payload: dict[str, Any], embedding: list[float] | None = None) -> Chunk:
        """Convert Qdrant payload to chunk.

        Args:
            payload: Qdrant payload dictionary
            embedding: Optional embedding vector

        Returns:
            Reconstructed chunk
        """
        metadata = ChunkMetadata(
            document_id=payload["document_id"],
            chunk_index=payload["chunk_index"],
            chunk_type=ChunkType(payload["chunk_type"]),
            start_char=payload.get("start_char"),
            end_char=payload.get("end_char"),
            page_numbers=payload.get("page_numbers", []),
            section_title=payload.get("section_title"),
            section_level=payload.get("section_level"),
            extra=payload.get("extra") or {},
        )

        return Chunk(
            id=payload["chunk_id"],
            content=payload["content"],
            embedding=embedding,
            metadata=metadata,
            validation_scores=payload.get("validation_scores", {}),
            parent_id=payload.get("parent_id"),
            children_ids=payload.get("children_ids", []),
        )

    async def upsert(self, chunks: list[Chunk]) -> None:
        """Insert or update chunks.

        Args:
            chunks: Chunks to upsert

        Raises:
            ValueError: If chunks don't have embeddings
        """
        await self.initialize()

        if not chunks:
            return

        # Validate embeddings
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk {chunk.id} does not have an embedding")

        # Prepare points
        points = []
        for chunk in chunks:
            point = qmodels.PointStruct(
                id=chunk.id,
                vector=chunk.embedding,
                payload=self._chunk_to_payload(chunk),
            )
            points.append(point)

        # Upsert in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            await self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )

        logger.info(f"Upserted {len(chunks)} chunks to Qdrant collection {self.collection_name}")

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
            filter_dict: Metadata filters (Qdrant filter format)

        Returns:
            List of (chunk, score) tuples
        """
        await self.initialize()

        # Build filter if provided
        query_filter = None
        if filter_dict:
            # Convert dict to Qdrant filter
            # Supports both single values and lists (for "any of" matching)
            must_conditions = []
            for key, value in filter_dict.items():
                if isinstance(value, list):
                    # Use MatchAny for list values (OR within the list)
                    must_conditions.append(
                        qmodels.FieldCondition(key=key, match=qmodels.MatchAny(any=value))
                    )
                else:
                    # Use MatchValue for single values
                    must_conditions.append(
                        qmodels.FieldCondition(key=key, match=qmodels.MatchValue(value=value))
                    )

            query_filter = qmodels.Filter(must=must_conditions)

        # Search/query using the new Qdrant API
        search_result = await self.client.query_points(
            collection_name=self.collection_name,
            query=embedding,
            limit=top_k,
            query_filter=query_filter,
        )

        # Convert results
        results = []
        for scored_point in search_result.points:
            chunk = self._payload_to_chunk(scored_point.payload, embedding=scored_point.vector)
            score = scored_point.score
            results.append((chunk, score))

        logger.debug(f"Query returned {len(results)} results")

        return results

    async def delete(self, chunk_ids: list[str]) -> None:
        """Delete chunks.

        Args:
            chunk_ids: IDs to delete
        """
        await self.initialize()

        await self.client.delete(
            collection_name=self.collection_name,
            points_selector=qmodels.PointIdsList(points=chunk_ids),
        )

        logger.debug(f"Deleted {len(chunk_ids)} chunks from Qdrant")

    async def delete_by_document_id(self, document_id: str) -> int:
        """Delete all chunks belonging to a document.

        Args:
            document_id: The document ID whose chunks should be deleted

        Returns:
            Number of chunks deleted (approximate)
        """
        await self.initialize()

        # Use filter to delete all chunks with this document_id
        delete_filter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="document_id",
                    match=qmodels.MatchValue(value=document_id),
                )
            ]
        )

        # Get count before delete (for logging)
        scroll_result = await self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=delete_filter,
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        # Note: scroll only returns first page, so count may be approximate for large sets

        await self.client.delete(
            collection_name=self.collection_name,
            points_selector=qmodels.FilterSelector(filter=delete_filter),
        )

        deleted_count = len(scroll_result[0]) if scroll_result[0] else 0
        logger.info(f"Deleted chunks for document {document_id} from Qdrant (at least {deleted_count})")

        return deleted_count

    async def get(self, chunk_ids: list[str]) -> list[Chunk]:
        """Get chunks by ID.

        Args:
            chunk_ids: IDs to retrieve

        Returns:
            List of chunks
        """
        await self.initialize()

        points = await self.client.retrieve(
            collection_name=self.collection_name,
            ids=chunk_ids,
            with_vectors=True,
        )

        chunks = []
        for point in points:
            chunk = self._payload_to_chunk(point.payload, embedding=point.vector)
            chunks.append(chunk)

        logger.debug(f"Retrieved {len(chunks)}/{len(chunk_ids)} chunks from Qdrant")

        return chunks

    async def count(self) -> int:
        """Get total chunk count.

        Returns:
            Number of chunks
        """
        await self.initialize()

        collection_info = await self.client.get_collection(self.collection_name)
        return collection_info.points_count

    async def get_by_document_id(self, document_id: str, limit: int = 1000) -> list[Chunk]:
        """Retrieve all chunks belonging to a document.

        Args:
            document_id: The document ID to get chunks for
            limit: Maximum number of chunks to return (default 1000)

        Returns:
            List of chunks sorted by chunk_index
        """
        await self.initialize()

        # Use scroll with filter to get all chunks for document
        doc_filter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="document_id",
                    match=qmodels.MatchValue(value=document_id),
                )
            ]
        )

        all_chunks = []
        offset = None

        while True:
            points, next_offset = await self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=doc_filter,
                limit=min(100, limit - len(all_chunks)),
                offset=offset,
                with_vectors=False,  # Don't need embeddings for display
                with_payload=True,
            )

            for point in points:
                chunk = self._payload_to_chunk(point.payload, embedding=None)
                all_chunks.append(chunk)

            if next_offset is None or len(all_chunks) >= limit:
                break
            offset = next_offset

        # Sort by chunk_index
        all_chunks.sort(key=lambda c: c.metadata.chunk_index)

        logger.debug(f"Retrieved {len(all_chunks)} chunks for document {document_id}")
        return all_chunks

    async def clear(self) -> None:
        """Clear all chunks."""
        await self.initialize()

        # Delete and recreate collection
        await self.client.delete_collection(self.collection_name)

        await self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qmodels.VectorParams(
                size=self.embedding_dimension,
                distance=self._get_distance_metric(),
            ),
        )

        logger.info(f"Cleared collection {self.collection_name}")

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
        await self.initialize()

        # Build filter for document_id and position range
        must_conditions = [
            qmodels.FieldCondition(
                key="document_id",
                match=qmodels.MatchValue(value=document_id),
            )
        ]

        # Add range condition based on position field
        if position_field == "chunk_index":
            must_conditions.append(
                qmodels.FieldCondition(
                    key="chunk_index",
                    range=qmodels.Range(
                        gte=position_start,
                        lte=position_end,
                    ),
                )
            )
        elif position_field == "page_number":
            # For page_numbers (list field), we need to check if any value is in range
            # Qdrant doesn't have built-in list intersection, so we scroll and filter
            pass

        query_filter = qmodels.Filter(must=must_conditions)

        # Use scroll to get all matching points
        points, _next_offset = await self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=query_filter,
            limit=1000,  # Reasonable limit for context window
            with_vectors=True,
            with_payload=True,
        )

        # Convert to chunks
        chunks = []
        for point in points:
            chunk = self._payload_to_chunk(point.payload, embedding=point.vector)
            
            # Additional filtering for page_number (since Qdrant doesn't support list range queries)
            if position_field == "page_number":
                if chunk.metadata.page_numbers:
                    if not any(position_start <= page <= position_end for page in chunk.metadata.page_numbers):
                        continue
                else:
                    continue
            
            chunks.append(chunk)

        # Sort by position
        if position_field == "chunk_index":
            chunks.sort(key=lambda c: c.metadata.chunk_index)
        elif position_field == "page_number":
            chunks.sort(key=lambda c: c.metadata.page_numbers[0] if c.metadata.page_numbers else 0)

        logger.debug(
            f"Retrieved {len(chunks)} chunks by {position_field} "
            f"range [{position_start}, {position_end}] for document {document_id}"
        )

        return chunks

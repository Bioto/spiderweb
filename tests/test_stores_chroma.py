"""Tests for Chroma vector store implementation.

Tests ChromaVectorStore interface with mocked chromadb client
and collection to verify integration points without real database.
"""

import sys

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from spiderweb.models.document import Chunk, ChunkMetadata, ChunkType
from spiderweb.stores.chroma import ChromaVectorStore


def _make_chunk(content: str, idx: int = 0, doc_id: str = "doc1", embedding: list[float] | None = None) -> Chunk:
    """Create a test chunk with embedding."""
    chunk = Chunk(
        content=content,
        metadata=ChunkMetadata(
            document_id=doc_id,
            chunk_index=idx,
            chunk_type=ChunkType.CUSTOM,
        ),
    )
    if embedding is None:
        embedding = [0.1] * 1536  # Default embedding
    chunk.embedding = embedding
    return chunk


def _make_chromadb_mock(mock_client, mock_collection):
    """Build a mock chromadb module so initialize() can import it."""
    mock_client.get_collection = MagicMock(side_effect=Exception("Not found"))
    mock_client.create_collection = MagicMock(return_value=mock_collection)
    mock_config = MagicMock()
    mock_config.Settings = MagicMock()
    mock_chromadb = MagicMock()
    mock_chromadb.PersistentClient = MagicMock(return_value=mock_client)
    mock_chromadb.Client = MagicMock(return_value=mock_client)
    mock_chromadb.config = mock_config
    return mock_chromadb, mock_config


class TestChromaVectorStoreInit:
    """Tests for ChromaVectorStore initialization."""

    def test_init_defaults(self):
        """Initializes with default parameters."""
        store = ChromaVectorStore()
        assert store.collection_name == "spiderweb_documents"
        assert store.persist_directory is None
        assert store.embedding_dimension == 1536
        assert not store._initialized

    def test_init_custom_params(self):
        """Initializes with custom parameters."""
        store = ChromaVectorStore(
            collection_name="custom_collection",
            persist_directory="/tmp/chroma",
            embedding_dimension=768,
        )
        assert store.collection_name == "custom_collection"
        assert store.persist_directory == "/tmp/chroma"
        assert store.embedding_dimension == 768


class TestChromaVectorStoreInitialize:
    """Tests for initialize method."""

    @pytest.mark.asyncio
    async def test_initialize_creates_persistent_client(self):
        """Initialize creates PersistentClient when persist_directory set."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_chromadb, mock_config = _make_chromadb_mock(mock_client, mock_collection)
        mock_chromadb.PersistentClient = MagicMock(return_value=mock_client)

        store = ChromaVectorStore(persist_directory="/tmp/chroma")

        with patch.dict(sys.modules, {"chromadb": mock_chromadb, "chromadb.config": mock_config}):
            await store.initialize()

        assert store._initialized
        assert store._client == mock_client
        assert store._collection == mock_collection

    @pytest.mark.asyncio
    async def test_initialize_creates_in_memory_client(self):
        """Initialize creates in-memory Client when persist_directory None."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_chromadb, mock_config = _make_chromadb_mock(mock_client, mock_collection)

        store = ChromaVectorStore(persist_directory=None)

        with patch.dict(sys.modules, {"chromadb": mock_chromadb, "chromadb.config": mock_config}):
            await store.initialize()

        assert store._initialized

    @pytest.mark.asyncio
    async def test_initialize_gets_existing_collection(self):
        """Initialize retrieves existing collection if it exists."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_chromadb, mock_config = _make_chromadb_mock(mock_client, mock_collection)
        mock_chromadb.Client = MagicMock(return_value=mock_client)
        # Override helper: get_collection succeeds so create_collection is not used
        mock_client.get_collection = MagicMock(return_value=mock_collection)

        store = ChromaVectorStore()

        with patch.dict(sys.modules, {"chromadb": mock_chromadb, "chromadb.config": mock_config}):
            await store.initialize()

        assert store._collection == mock_collection
        mock_client.create_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_initialize_missing_chromadb(self):
        """Initialize raises ImportError when chromadb not installed."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "chromadb":
                raise ImportError("No module named 'chromadb'")
            return real_import(name, *args, **kwargs)

        store = ChromaVectorStore()

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="chromadb is not installed"):
                await store.initialize()

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self):
        """Multiple initialize calls are idempotent."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_chromadb, mock_config = _make_chromadb_mock(mock_client, mock_collection)

        store = ChromaVectorStore()

        with patch.dict(sys.modules, {"chromadb": mock_chromadb, "chromadb.config": mock_config}):
            await store.initialize()
            await store.initialize()  # Second call should not create new collection

        assert mock_client.create_collection.call_count == 1


class TestChromaVectorStoreUpsert:
    """Tests for upsert method."""

    @pytest.mark.asyncio
    async def test_upsert_stores_chunks(self):
        """Upsert stores chunks in Chroma collection."""
        mock_collection = MagicMock()
        mock_collection.upsert = MagicMock()

        store = ChromaVectorStore()
        store._collection = mock_collection
        store._initialized = True

        chunks = [
            _make_chunk("Content 1", idx=0),
            _make_chunk("Content 2", idx=1),
        ]

        await store.upsert(chunks)

        mock_collection.upsert.assert_called_once()
        call_kwargs = mock_collection.upsert.call_args[1]
        assert len(call_kwargs["ids"]) == 2
        assert len(call_kwargs["embeddings"]) == 2
        assert len(call_kwargs["documents"]) == 2
        assert len(call_kwargs["metadatas"]) == 2

    @pytest.mark.asyncio
    async def test_upsert_requires_embeddings(self):
        """Upsert raises ValueError for chunks without embeddings."""
        store = ChromaVectorStore()
        store._collection = MagicMock()
        store._initialized = True

        chunk = _make_chunk("Content")
        chunk.embedding = None  # Override so chunk has no embedding

        with pytest.raises(ValueError, match="does not have an embedding"):
            await store.upsert([chunk])

    @pytest.mark.asyncio
    async def test_upsert_empty_list(self):
        """Upsert handles empty chunk list."""
        store = ChromaVectorStore()
        store._collection = MagicMock()
        store._initialized = True

        await store.upsert([])

        store._collection.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_upsert_auto_initializes(self):
        """Upsert auto-initializes if not initialized."""
        mock_collection = MagicMock()
        mock_collection.upsert = MagicMock()
        mock_client = MagicMock()
        mock_chromadb, mock_config = _make_chromadb_mock(mock_client, mock_collection)

        store = ChromaVectorStore()

        with patch.dict(sys.modules, {"chromadb": mock_chromadb, "chromadb.config": mock_config}):
            await store.upsert([_make_chunk("Content")])

        assert store._initialized
        mock_collection.upsert.assert_called_once()


class TestChromaVectorStoreQuery:
    """Tests for query method."""

    @pytest.mark.asyncio
    async def test_query_returns_chunks_with_scores(self):
        """Query returns chunks with similarity scores."""
        mock_collection = MagicMock()
        mock_collection.query = MagicMock(return_value={
            "ids": [["chunk1", "chunk2"]],
            "documents": [["Content 1", "Content 2"]],
            "metadatas": [[
                {"document_id": "doc1", "chunk_index": 0, "chunk_type": "custom"},
                {"document_id": "doc1", "chunk_index": 1, "chunk_type": "custom"},
            ]],
            "embeddings": [[[0.1] * 1536, [0.2] * 1536]],
            "distances": [[0.1, 0.2]],
        })

        store = ChromaVectorStore()
        store._collection = mock_collection
        store._initialized = True

        query_embedding = [0.1] * 1536
        results = await store.query(query_embedding, top_k=2)

        assert len(results) == 2
        assert isinstance(results[0][0], Chunk)
        assert results[0][0].content == "Content 1"
        # Distance 0.1 -> similarity 0.9
        assert results[0][1] == pytest.approx(0.9, abs=0.01)

    @pytest.mark.asyncio
    async def test_query_applies_top_k(self):
        """Query respects top_k parameter."""
        mock_collection = MagicMock()
        mock_collection.query = MagicMock(return_value={
            "ids": [["chunk1"]],
            "documents": [["Content"]],
            "metadatas": [[{"document_id": "doc1", "chunk_index": 0, "chunk_type": "custom"}]],
            "embeddings": [[[0.1] * 1536]],
            "distances": [[0.1]],
        })

        store = ChromaVectorStore()
        store._collection = mock_collection
        store._initialized = True

        await store.query([0.1] * 1536, top_k=5)

        call_kwargs = mock_collection.query.call_args[1]
        assert call_kwargs["n_results"] == 5

    @pytest.mark.asyncio
    async def test_query_with_filter_dict(self):
        """Query applies filter_dict as Chroma where clause."""
        mock_collection = MagicMock()
        mock_collection.query = MagicMock(return_value={
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "embeddings": [[]],
            "distances": [[]],
        })

        store = ChromaVectorStore()
        store._collection = mock_collection
        store._initialized = True

        await store.query([0.1] * 1536, filter_dict={"document_id": "doc1"})

        call_kwargs = mock_collection.query.call_args[1]
        assert call_kwargs["where"] == {"document_id": {"$eq": "doc1"}}

    @pytest.mark.asyncio
    async def test_query_filter_dict_list(self):
        """Query converts list filter to $in clause."""
        mock_collection = MagicMock()
        mock_collection.query = MagicMock(return_value={
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "embeddings": [[]],
            "distances": [[]],
        })

        store = ChromaVectorStore()
        store._collection = mock_collection
        store._initialized = True

        await store.query([0.1] * 1536, filter_dict={"document_id": ["doc1", "doc2"]})

        call_kwargs = mock_collection.query.call_args[1]
        assert call_kwargs["where"] == {"document_id": {"$in": ["doc1", "doc2"]}}

    @pytest.mark.asyncio
    async def test_query_empty_results(self):
        """Query handles empty results gracefully."""
        mock_collection = MagicMock()
        mock_collection.query = MagicMock(return_value={
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "embeddings": [[]],
            "distances": [[]],
        })

        store = ChromaVectorStore()
        store._collection = mock_collection
        store._initialized = True

        results = await store.query([0.1] * 1536)

        assert results == []


class TestChromaVectorStoreDelete:
    """Tests for delete methods."""

    @pytest.mark.asyncio
    async def test_delete_chunks(self):
        """Delete removes chunks by IDs."""
        mock_collection = MagicMock()
        mock_collection.delete = MagicMock()

        store = ChromaVectorStore()
        store._collection = mock_collection
        store._initialized = True

        await store.delete(["chunk1", "chunk2"])

        mock_collection.delete.assert_called_once_with(ids=["chunk1", "chunk2"])

    @pytest.mark.asyncio
    async def test_delete_by_document_id(self):
        """delete_by_document_id removes all chunks for document."""
        mock_collection = MagicMock()
        mock_collection.get = MagicMock(return_value={
            "ids": ["chunk1", "chunk2"],
            "documents": ["Content 1", "Content 2"],
            "metadatas": [
                {"document_id": "doc1", "chunk_index": 0, "chunk_type": "custom"},
                {"document_id": "doc1", "chunk_index": 1, "chunk_type": "custom"},
            ],
        })
        mock_collection.delete = MagicMock()

        store = ChromaVectorStore()
        store._collection = mock_collection
        store._initialized = True

        deleted_count = await store.delete_by_document_id("doc1")

        assert deleted_count == 2
        mock_collection.delete.assert_called_once_with(ids=["chunk1", "chunk2"])

    @pytest.mark.asyncio
    async def test_delete_by_document_id_no_matches(self):
        """delete_by_document_id returns 0 when no chunks found."""
        mock_collection = MagicMock()
        mock_collection.get = MagicMock(return_value={
            "ids": [],
            "documents": [],
            "metadatas": [],
        })

        store = ChromaVectorStore()
        store._collection = mock_collection
        store._initialized = True

        deleted_count = await store.delete_by_document_id("nonexistent")

        assert deleted_count == 0


class TestChromaVectorStoreOther:
    """Tests for other ChromaVectorStore methods."""

    @pytest.mark.asyncio
    async def test_get_chunks(self):
        """get retrieves chunks by IDs."""
        mock_collection = MagicMock()
        mock_collection.get = MagicMock(return_value={
            "ids": ["chunk1"],
            "documents": ["Content"],
            "metadatas": [{"document_id": "doc1", "chunk_index": 0, "chunk_type": "custom"}],
            "embeddings": [[0.1] * 1536],
        })

        store = ChromaVectorStore()
        store._collection = mock_collection
        store._initialized = True

        chunks = await store.get(["chunk1"])

        assert len(chunks) == 1
        assert isinstance(chunks[0], Chunk)
        assert chunks[0].id == "chunk1"

    @pytest.mark.asyncio
    async def test_count(self):
        """count returns chunk count."""
        mock_collection = MagicMock()
        mock_collection.count = MagicMock(return_value=42)

        store = ChromaVectorStore()
        store._collection = mock_collection
        store._initialized = True

        count = await store.count()

        assert count == 42

    @pytest.mark.asyncio
    async def test_clear(self):
        """clear deletes and recreates collection."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.delete_collection = MagicMock()
        mock_client.create_collection = MagicMock(return_value=mock_collection)

        store = ChromaVectorStore()
        store._client = mock_client
        store._collection = mock_collection
        store._initialized = True

        await store.clear()

        mock_client.delete_collection.assert_called_once_with(name=store.collection_name)
        mock_client.create_collection.assert_called_once()

"""Tests for vector store implementations."""

import pytest

from spiderweb.models.document import Chunk, ChunkMetadata, ChunkType
from spiderweb.stores.memory import MemoryVectorStore


def _make_chunk(
    content: str,
    *,
    idx: int,
    embedding: list[float] | None,
    doc_id: str = "doc",
) -> Chunk:
    chunk = Chunk(
        content=content,
        metadata=ChunkMetadata(
            document_id=doc_id,
            chunk_index=idx,
            chunk_type=ChunkType.CUSTOM,
        ),
    )
    chunk.embedding = embedding
    return chunk


async def test_memory_vector_store_upsert_requires_embeddings():
    store = MemoryVectorStore()
    chunk = _make_chunk("Hello", idx=0, embedding=None)

    with pytest.raises(ValueError, match="does not have an embedding"):
        await store.upsert([chunk])


async def test_memory_vector_store_query_orders_by_similarity():
    store = MemoryVectorStore()
    chunk_a = _make_chunk("A", idx=0, embedding=[1.0, 0.0])
    chunk_b = _make_chunk("B", idx=1, embedding=[0.0, 1.0])
    await store.upsert([chunk_a, chunk_b])

    results = await store.query([1.0, 0.0], top_k=2)

    assert len(results) == 2
    assert results[0][0].id == chunk_a.id
    assert results[0][1] == pytest.approx(1.0, abs=1e-6)


class TestMemoryVectorStoreFilterDict:
    """Tests for filter_dict functionality in MemoryVectorStore."""

    def _make_chunk_with_metadata(
        self,
        content: str,
        *,
        idx: int,
        embedding: list[float],
        doc_id: str = "doc",
        extra: dict | None = None,
    ) -> Chunk:
        """Create chunk with custom metadata."""
        chunk = Chunk(
            content=content,
            metadata=ChunkMetadata(
                document_id=doc_id,
                chunk_index=idx,
                chunk_type=ChunkType.CUSTOM,
                extra=extra or {},
            ),
        )
        chunk.embedding = embedding
        return chunk

    async def test_filter_dict_exact_match_document_id(self):
        """Filter by document_id returns only matching chunks."""
        store = MemoryVectorStore()
        chunk1 = self._make_chunk_with_metadata("Doc 1", idx=0, embedding=[1.0, 0.0], doc_id="doc1")
        chunk2 = self._make_chunk_with_metadata("Doc 2", idx=0, embedding=[1.0, 0.0], doc_id="doc2")
        chunk3 = self._make_chunk_with_metadata("Doc 1 again", idx=1, embedding=[1.0, 0.0], doc_id="doc1")
        await store.upsert([chunk1, chunk2, chunk3])

        results = await store.query([1.0, 0.0], top_k=10, filter_dict={"document_id": "doc1"})

        assert len(results) == 2
        assert all(chunk.metadata.document_id == "doc1" for chunk, _ in results)

    async def test_filter_dict_list_any_of(self):
        """Filter with list value matches any value in list."""
        store = MemoryVectorStore()
        chunk1 = self._make_chunk_with_metadata("A", idx=0, embedding=[1.0, 0.0], doc_id="doc1")
        chunk2 = self._make_chunk_with_metadata("B", idx=0, embedding=[1.0, 0.0], doc_id="doc2")
        chunk3 = self._make_chunk_with_metadata("C", idx=0, embedding=[1.0, 0.0], doc_id="doc3")
        await store.upsert([chunk1, chunk2, chunk3])

        results = await store.query([1.0, 0.0], top_k=10, filter_dict={"document_id": ["doc1", "doc3"]})

        assert len(results) == 2
        doc_ids = {chunk.metadata.document_id for chunk, _ in results}
        assert doc_ids == {"doc1", "doc3"}

    async def test_filter_dict_extra_metadata(self):
        """Filter by extra metadata dict works."""
        store = MemoryVectorStore()
        chunk1 = self._make_chunk_with_metadata(
            "A", idx=0, embedding=[1.0, 0.0], extra={"category": "tech"}
        )
        chunk2 = self._make_chunk_with_metadata(
            "B", idx=0, embedding=[1.0, 0.0], extra={"category": "science"}
        )
        chunk3 = self._make_chunk_with_metadata(
            "C", idx=0, embedding=[1.0, 0.0], extra={"category": "tech"}
        )
        await store.upsert([chunk1, chunk2, chunk3])

        results = await store.query([1.0, 0.0], top_k=10, filter_dict={"category": "tech"})

        assert len(results) == 2
        assert all(chunk.metadata.extra.get("category") == "tech" for chunk, _ in results)

    async def test_filter_dict_extra_list(self):
        """Filter extra metadata with list value."""
        store = MemoryVectorStore()
        chunk1 = self._make_chunk_with_metadata(
            "A", idx=0, embedding=[1.0, 0.0], extra={"tag": "python"}
        )
        chunk2 = self._make_chunk_with_metadata(
            "B", idx=0, embedding=[1.0, 0.0], extra={"tag": "javascript"}
        )
        chunk3 = self._make_chunk_with_metadata(
            "C", idx=0, embedding=[1.0, 0.0], extra={"tag": "rust"}
        )
        await store.upsert([chunk1, chunk2, chunk3])

        results = await store.query([1.0, 0.0], top_k=10, filter_dict={"tag": ["python", "rust"]})

        assert len(results) == 2
        tags = {chunk.metadata.extra.get("tag") for chunk, _ in results}
        assert tags == {"python", "rust"}

    async def test_filter_dict_no_match(self):
        """Filter that matches no chunks returns empty list."""
        store = MemoryVectorStore()
        chunk1 = self._make_chunk_with_metadata("A", idx=0, embedding=[1.0, 0.0], doc_id="doc1")
        chunk2 = self._make_chunk_with_metadata("B", idx=0, embedding=[1.0, 0.0], doc_id="doc2")
        await store.upsert([chunk1, chunk2])

        results = await store.query([1.0, 0.0], top_k=10, filter_dict={"document_id": "doc999"})

        assert len(results) == 0

    async def test_filter_dict_none_no_filter(self):
        """filter_dict=None behaves same as no filter."""
        store = MemoryVectorStore()
        chunk1 = _make_chunk("A", idx=0, embedding=[1.0, 0.0], doc_id="doc1")
        chunk2 = _make_chunk("B", idx=0, embedding=[1.0, 0.0], doc_id="doc2")
        await store.upsert([chunk1, chunk2])

        results_with_filter = await store.query([1.0, 0.0], top_k=10, filter_dict=None)
        results_without_filter = await store.query([1.0, 0.0], top_k=10)

        assert len(results_with_filter) == len(results_without_filter) == 2

    async def test_filter_dict_multiple_keys(self):
        """Filter with multiple keys requires all to match."""
        store = MemoryVectorStore()
        chunk1 = self._make_chunk_with_metadata(
            "A", idx=0, embedding=[1.0, 0.0], doc_id="doc1", extra={"category": "tech"}
        )
        chunk2 = self._make_chunk_with_metadata(
            "B", idx=0, embedding=[1.0, 0.0], doc_id="doc1", extra={"category": "science"}
        )
        chunk3 = self._make_chunk_with_metadata(
            "C", idx=0, embedding=[1.0, 0.0], doc_id="doc2", extra={"category": "tech"}
        )
        await store.upsert([chunk1, chunk2, chunk3])

        results = await store.query(
            [1.0, 0.0],
            top_k=10,
            filter_dict={"document_id": "doc1", "category": "tech"},
        )

        assert len(results) == 1
        assert results[0][0].metadata.document_id == "doc1"
        assert results[0][0].metadata.extra.get("category") == "tech"

    async def test_filter_dict_missing_key(self):
        """Filter by key that doesn't exist in metadata excludes chunk."""
        store = MemoryVectorStore()
        chunk1 = self._make_chunk_with_metadata("A", idx=0, embedding=[1.0, 0.0], extra={"key1": "value1"})
        chunk2 = self._make_chunk_with_metadata("B", idx=0, embedding=[1.0, 0.0], extra={"key2": "value2"})
        await store.upsert([chunk1, chunk2])

        results = await store.query([1.0, 0.0], top_k=10, filter_dict={"nonexistent": "value"})

        assert len(results) == 0




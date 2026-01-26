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




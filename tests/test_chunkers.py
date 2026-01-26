"""Tests for chunker components."""

import pytest

from spiderweb.chunkers.hierarchical import HierarchicalChunker
from spiderweb.chunkers.sentence import SentenceChunker
from spiderweb.chunkers.sliding_window import SlidingWindowChunker
from spiderweb.models.document import ChunkMetadata, ChunkType, Document, DocumentMetadata


def _make_document(markdown: str) -> Document:
    return Document(
        raw_content=markdown,
        markdown_content=markdown,
        metadata=DocumentMetadata(
            source="memory://test",
            file_type="md",
            extraction_method="test",
        ),
    )


def test_hierarchical_chunker_creates_parent_child_relationships():
    doc = _make_document(
        "# Title\n\nIntro text.\n\n## Section A\n\nDetails A.\n\n## Section B\n\nDetails B.\n"
    )

    chunker = HierarchicalChunker(max_chunk_size=10_000)
    chunks = chunker.chunk(doc)

    assert len(chunks) >= 3
    assert chunks[0].metadata.chunk_type == ChunkType.HIERARCHICAL

    # First chunk is a top-level heading (# -> level 1), so no parent.
    assert chunks[0].parent_id is None

    # Section chunks should be children of the most recent higher-level chunk.
    section_a = next(c for c in chunks if c.metadata.section_title == "Section A")
    assert section_a.parent_id == chunks[0].id
    assert section_a.id in chunks[0].children_ids


def test_sentence_chunker_splits_and_sets_metadata():
    doc = _make_document("One. Two. Three. Four. Five.")
    chunker = SentenceChunker(max_chunk_size=10, min_chunk_size=1)
    chunks = chunker.chunk(doc)

    assert len(chunks) >= 2
    assert [c.metadata.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(c.metadata.chunk_type == ChunkType.SENTENCE for c in chunks)


def test_sliding_window_chunker_simple_overlap():
    doc = _make_document("abcdefghij0123456789")  # 20 chars
    chunker = SlidingWindowChunker(max_chunk_size=10, chunk_overlap=2, respect_boundaries=False)
    chunks = chunker.chunk(doc)

    assert [c.content for c in chunks] == ["abcdefghij", "ij01234567", "6789"]
    assert [c.metadata.chunk_index for c in chunks] == [0, 1, 2]
    assert all(c.metadata.chunk_type == ChunkType.SLIDING_WINDOW for c in chunks)


def test_chunkers_handle_empty_documents():
    doc = _make_document("")
    assert HierarchicalChunker().chunk(doc) == []
    assert SentenceChunker().chunk(doc) == []
    assert SlidingWindowChunker().chunk(doc) == []



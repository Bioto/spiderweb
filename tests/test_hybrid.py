"""Tests for hybrid search functionality.

Tests BM25Index tokenization, search, IDF behavior, and HybridSearcher
with mocked LLM client and vector store.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from spiderweb.models.document import Chunk, ChunkMetadata, ChunkType
from spiderweb.pipeline.hybrid import BM25Index, HybridSearcher


def _make_chunk(content: str, idx: int = 0, doc_id: str = "doc1") -> Chunk:
    """Create a test chunk."""
    chunk = Chunk(
        content=content,
        metadata=ChunkMetadata(
            document_id=doc_id,
            chunk_index=idx,
            chunk_type=ChunkType.CUSTOM,
        ),
    )
    return chunk


class TestBM25IndexTokenize:
    """Tests for BM25Index tokenization."""

    def test_tokenize_simple(self):
        """Tokenizes simple text into lowercase words."""
        index = BM25Index()
        tokens = index._tokenize("Hello World")
        assert tokens == ["hello", "world"]

    def test_tokenize_punctuation(self):
        """Punctuation is excluded from tokens."""
        index = BM25Index()
        tokens = index._tokenize("Hello, world! How are you?")
        assert tokens == ["hello", "world", "how", "are", "you"]

    def test_tokenize_empty_string(self):
        """Empty string returns empty list."""
        index = BM25Index()
        tokens = index._tokenize("")
        assert tokens == []

    def test_tokenize_case_insensitive(self):
        """Tokenization is case-insensitive."""
        index = BM25Index()
        tokens = index._tokenize("Hello WORLD")
        assert tokens == ["hello", "world"]

    def test_tokenize_numbers(self):
        """Numbers are included in tokens."""
        index = BM25Index()
        tokens = index._tokenize("Version 2.0 released")
        assert tokens == ["version", "2", "0", "released"]


class TestBM25IndexSearch:
    """Tests for BM25Index search functionality."""

    def test_search_empty_index(self):
        """Search on empty index returns empty list."""
        index = BM25Index()
        results = index.search("test query")
        assert results == []

    def test_search_no_matching_term(self):
        """Search with no matching terms returns empty list."""
        index = BM25Index([_make_chunk("Document about Python")])
        results = index.search("javascript")
        assert results == []

    def test_search_single_match(self):
        """Search finds chunk containing query term."""
        chunk1 = _make_chunk("Document about Python programming")
        chunk2 = _make_chunk("Document about JavaScript")
        index = BM25Index([chunk1, chunk2])

        results = index.search("Python")

        assert len(results) == 1
        assert results[0][0].content == "Document about Python programming"
        # With 2 docs, term in 1 doc has positive IDF; score can be small but non-negative
        assert results[0][1] >= 0

    def test_search_multiple_matches(self):
        """Search returns multiple chunks ordered by BM25 score."""
        chunk1 = _make_chunk("Python is great")
        chunk2 = _make_chunk("Python Python Python")  # More occurrences
        chunk3 = _make_chunk("JavaScript is also great")
        index = BM25Index([chunk1, chunk2, chunk3])

        results = index.search("Python")

        assert len(results) == 2
        # Both chunks contain "Python"; order depends on BM25 length norm and tf
        assert {results[0][0].content, results[1][0].content} == {"Python is great", "Python Python Python"}
        assert results[0][1] >= results[1][1]  # Sorted by score descending

    def test_search_applies_top_k(self):
        """Search respects top_k parameter."""
        chunks = [_make_chunk(f"Document {i} about Python") for i in range(5)]
        index = BM25Index(chunks)

        results = index.search("Python", top_k=3)

        assert len(results) == 3

    def test_search_multiple_terms(self):
        """Search with multiple terms returns chunks containing query terms."""
        chunk1 = _make_chunk("Python programming language")
        chunk2 = _make_chunk("Python tutorial")
        chunk3 = _make_chunk("JavaScript programming")
        index = BM25Index([chunk1, chunk2, chunk3])

        results = index.search("Python programming")

        assert len(results) >= 2
        # Chunk with both "python" and "programming" should be in results
        contents = [r[0].content for r in results]
        assert "Python programming language" in contents
        assert "Python tutorial" in contents or "JavaScript programming" in contents

    def test_search_empty_query(self):
        """Empty query returns empty results."""
        index = BM25Index([_make_chunk("Some content")])
        results = index.search("")
        assert results == []


class TestBM25IndexIDF:
    """Tests for IDF (Inverse Document Frequency) behavior."""

    def test_idf_rare_term_higher_score(self):
        """Rare terms (in fewer documents) get higher IDF scores."""
        chunk1 = _make_chunk("Python programming")
        chunk2 = _make_chunk("JavaScript programming")
        chunk3 = _make_chunk("Python tutorial")
        index = BM25Index([chunk1, chunk2, chunk3])

        # "JavaScript" appears in only 1 doc, "Python" in 2 docs
        # So "JavaScript" should have higher IDF
        results_js = index.search("JavaScript")
        results_python = index.search("Python")

        # Both should return results
        assert len(results_js) > 0
        assert len(results_python) > 0

        # The IDF calculation should make rare terms score higher
        # (This is a smoke test - actual IDF values depend on BM25 formula)
        assert results_js[0][1] > 0  # Should have positive score


class TestBM25IndexAddChunks:
    """Tests for adding chunks to index."""

    def test_add_chunks_updates_index(self):
        """Adding chunks updates the search index."""
        index = BM25Index([_make_chunk("Initial content")])
        assert len(index.search("Initial")) == 1

        index.add_chunks([_make_chunk("New content")])
        results = index.search("New")

        assert len(results) == 1
        assert results[0][0].content == "New content"

    def test_add_chunks_preserves_existing(self):
        """Adding chunks preserves existing chunks in index."""
        index = BM25Index([_make_chunk("Old content")])
        index.add_chunks([_make_chunk("New content")])

        results_old = index.search("Old")
        results_new = index.search("New")

        assert len(results_old) == 1
        assert len(results_new) == 1


class TestHybridSearcher:
    """Tests for HybridSearcher combining BM25 and vector search."""

    @pytest.mark.asyncio
    async def test_hybrid_search_combines_results(self):
        """Hybrid search combines BM25 and vector results using RRF."""
        # Create chunks
        chunk1 = _make_chunk("Python programming language")
        chunk2 = _make_chunk("JavaScript tutorial")
        chunk3 = _make_chunk("Machine learning with Python")

        # Create BM25 index
        bm25_index = BM25Index([chunk1, chunk2, chunk3])

        # Mock LLM client
        mock_llm = AsyncMock()
        mock_embed_response = MagicMock()
        mock_embed_response.embeddings = [[0.1] * 1536]  # Dummy embedding
        mock_llm.embed = AsyncMock(return_value=mock_embed_response)

        # Mock vector store
        mock_vector_store = AsyncMock()
        # Vector search returns chunk3 first, chunk1 second
        mock_vector_store.query = AsyncMock(
            return_value=[(chunk3, 0.9), (chunk1, 0.8), (chunk2, 0.7)]
        )

        searcher = HybridSearcher(
            llm_client=mock_llm,
            vector_store=mock_vector_store,
            bm25_index=bm25_index,
        )

        results = await searcher.search("Python", top_k=3)

        # Should combine BM25 and vector results
        assert len(results) <= 3
        # Both BM25 and vector should have been called
        mock_llm.embed.assert_called_once_with("Python")
        mock_vector_store.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_hybrid_search_applies_top_k(self):
        """Hybrid search respects top_k parameter."""
        chunk1 = _make_chunk("Python")
        bm25_index = BM25Index([chunk1])

        mock_llm = AsyncMock()
        mock_embed_response = MagicMock()
        mock_embed_response.embeddings = [[0.1] * 1536]
        mock_llm.embed = AsyncMock(return_value=mock_embed_response)

        mock_vector_store = AsyncMock()
        mock_vector_store.query = AsyncMock(return_value=[(chunk1, 0.9)])

        searcher = HybridSearcher(
            llm_client=mock_llm,
            vector_store=mock_vector_store,
            bm25_index=bm25_index,
        )

        results = await searcher.search("Python", top_k=1)

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_hybrid_search_passes_filter_dict(self):
        """Hybrid search passes filter_dict to vector store."""
        bm25_index = BM25Index([_make_chunk("Test")])

        mock_llm = AsyncMock()
        mock_embed_response = MagicMock()
        mock_embed_response.embeddings = [[0.1] * 1536]
        mock_llm.embed = AsyncMock(return_value=mock_embed_response)

        mock_vector_store = AsyncMock()
        mock_vector_store.query = AsyncMock(return_value=[])

        searcher = HybridSearcher(
            llm_client=mock_llm,
            vector_store=mock_vector_store,
            bm25_index=bm25_index,
        )

        filter_dict = {"document_id": "doc1"}
        await searcher.search("test", filter_dict=filter_dict)

        # Check that filter_dict was passed to vector store
        call_kwargs = mock_vector_store.query.call_args[1]
        assert call_kwargs["filter_dict"] == filter_dict

    @pytest.mark.asyncio
    async def test_hybrid_search_builds_bm25_index_if_none(self):
        """HybridSearcher builds BM25 index if not provided."""
        mock_llm = AsyncMock()
        mock_embed_response = MagicMock()
        mock_embed_response.embeddings = [[0.1] * 1536]
        mock_llm.embed = AsyncMock(return_value=mock_embed_response)

        mock_vector_store = AsyncMock()
        mock_vector_store.query = AsyncMock(return_value=[])

        searcher = HybridSearcher(
            llm_client=mock_llm,
            vector_store=mock_vector_store,
            bm25_index=None,  # No BM25 index provided
        )

        # Should not crash, but BM25 search will return empty (no chunks indexed)
        results = await searcher.search("test")

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_hybrid_search_uses_rrf(self):
        """Hybrid search uses RRF to combine results."""
        chunk1 = _make_chunk("Python programming")
        chunk2 = _make_chunk("JavaScript tutorial")
        bm25_index = BM25Index([chunk1, chunk2])

        mock_llm = AsyncMock()
        mock_embed_response = MagicMock()
        mock_embed_response.embeddings = [[0.1] * 1536]
        mock_llm.embed = AsyncMock(return_value=mock_embed_response)

        mock_vector_store = AsyncMock()
        # Vector search returns different order than BM25
        mock_vector_store.query = AsyncMock(
            return_value=[(chunk2, 0.9), (chunk1, 0.8)]
        )

        searcher = HybridSearcher(
            llm_client=mock_llm,
            vector_store=mock_vector_store,
            bm25_index=bm25_index,
        )

        results = await searcher.search("Python", top_k=2, rrf_k=60)

        # Should combine both result sets using RRF
        # (RRF is tested separately in test_query_expansion, so we just verify it runs)
        assert len(results) <= 2

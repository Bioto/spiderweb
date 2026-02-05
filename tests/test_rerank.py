"""Tests for reranker functionality.

Tests reranking with different models (none, cross-encoder, cohere)
and error handling for missing dependencies.
"""

import sys

import pytest
from unittest.mock import MagicMock, patch

from spiderweb.models.document import Chunk, ChunkMetadata, ChunkType
from spiderweb.pipeline.rerank import Reranker


def _make_chunk(content: str, idx: int = 0) -> Chunk:
    """Create a test chunk."""
    chunk = Chunk(
        content=content,
        metadata=ChunkMetadata(
            document_id="doc1",
            chunk_index=idx,
            chunk_type=ChunkType.CUSTOM,
        ),
    )
    return chunk


class TestRerankerModelNone:
    """Tests for reranker with model='none'."""

    @pytest.mark.asyncio
    async def test_rerank_none_returns_unchanged(self):
        """model='none' returns results unchanged."""
        reranker = Reranker(model="none")
        chunks = [_make_chunk("A"), _make_chunk("B")]
        results = [(chunks[0], 0.9), (chunks[1], 0.7)]

        reranked = await reranker.rerank("test query", results)

        assert reranked == results

    @pytest.mark.asyncio
    async def test_rerank_none_applies_top_k(self):
        """model='none' applies top_k limit."""
        reranker = Reranker(model="none")
        chunks = [_make_chunk("A"), _make_chunk("B"), _make_chunk("C")]
        results = [(chunks[0], 0.9), (chunks[1], 0.7), (chunks[2], 0.5)]

        reranked = await reranker.rerank("test query", results, top_k=2)

        assert len(reranked) == 2
        assert reranked == results[:2]

    @pytest.mark.asyncio
    async def test_rerank_none_empty_results(self):
        """model='none' handles empty results."""
        reranker = Reranker(model="none")
        reranked = await reranker.rerank("test query", [])

        assert reranked == []


class TestRerankerCrossEncoder:
    """Tests for cross-encoder reranking."""

    @pytest.mark.asyncio
    async def test_rerank_cross_encoder_mocked(self):
        """Cross-encoder reranking with mocked sentence-transformers."""
        reranker = Reranker(model="cross-encoder")
        chunks = [_make_chunk("Document A"), _make_chunk("Document B")]
        results = [(chunks[0], 0.7), (chunks[1], 0.9)]  # B has higher vector score

        # Mock sentence_transformers module so we don't need it installed
        mock_cross_encoder = MagicMock()
        mock_cross_encoder.predict = MagicMock(return_value=[0.8, 0.6])
        mock_st = MagicMock()
        mock_st.CrossEncoder = MagicMock(return_value=mock_cross_encoder)

        with patch.dict(sys.modules, {"sentence_transformers": mock_st}):
            reranked = await reranker.rerank("test query", results)

        # Should be reordered: A first (0.8), B second (0.6)
        assert len(reranked) == 2
        assert reranked[0][0].content == "Document A"
        assert reranked[0][1] == 0.8
        assert reranked[1][0].content == "Document B"
        assert reranked[1][1] == 0.6

    @pytest.mark.asyncio
    async def test_rerank_cross_encoder_applies_top_k(self):
        """Cross-encoder applies top_k limit."""
        reranker = Reranker(model="cross-encoder")
        chunks = [_make_chunk("A"), _make_chunk("B"), _make_chunk("C")]
        results = [(chunks[0], 0.7), (chunks[1], 0.9), (chunks[2], 0.5)]

        mock_cross_encoder = MagicMock()
        mock_cross_encoder.predict = MagicMock(return_value=[0.8, 0.6, 0.4])
        mock_st = MagicMock()
        mock_st.CrossEncoder = MagicMock(return_value=mock_cross_encoder)

        with patch.dict(sys.modules, {"sentence_transformers": mock_st}):
            reranked = await reranker.rerank("test query", results, top_k=2)

        assert len(reranked) == 2

    @pytest.mark.asyncio
    async def test_rerank_cross_encoder_missing_dependency(self):
        """Cross-encoder falls back when sentence-transformers not installed."""
        reranker = Reranker(model="cross-encoder")
        chunks = [_make_chunk("A"), _make_chunk("B")]
        results = [(chunks[0], 0.9), (chunks[1], 0.7)]

        # Simulate sentence_transformers not being installed
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "sentence_transformers":
                raise ImportError("No module")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            reranked = await reranker.rerank("test query", results)

        # Should return original results unchanged
        assert reranked == results

    @pytest.mark.asyncio
    async def test_rerank_cross_encoder_custom_model_name(self):
        """Cross-encoder uses custom model_name from kwargs."""
        reranker = Reranker(model="cross-encoder", model_name="custom-model")
        chunks = [_make_chunk("A")]
        results = [(chunks[0], 0.9)]

        mock_cross_encoder = MagicMock()
        mock_cross_encoder.predict = MagicMock(return_value=[0.8])
        mock_st = MagicMock()
        mock_ce_class = MagicMock(return_value=mock_cross_encoder)
        mock_st.CrossEncoder = mock_ce_class

        with patch.dict(sys.modules, {"sentence_transformers": mock_st}):
            await reranker.rerank("test query", results)

        # Should be called with custom model name
        mock_ce_class.assert_called_once_with("custom-model")


class TestRerankerCohere:
    """Tests for Cohere reranking."""

    @pytest.mark.asyncio
    async def test_rerank_cohere_mocked(self):
        """Cohere reranking with mocked client."""
        mock_llm = MagicMock()
        reranker = Reranker(llm_client=mock_llm, model="cohere")
        chunks = [_make_chunk("Document A"), _make_chunk("Document B")]
        results = [(chunks[0], 0.7), (chunks[1], 0.9)]

        # Mock Cohere response
        mock_result1 = MagicMock()
        mock_result1.index = 1  # B comes first
        mock_result1.relevance_score = 0.95
        mock_result2 = MagicMock()
        mock_result2.index = 0  # A comes second
        mock_result2.relevance_score = 0.85

        mock_rerank_response = MagicMock()
        mock_rerank_response.results = [mock_result1, mock_result2]

        mock_cohere_client = MagicMock()
        mock_cohere_client.rerank = MagicMock(return_value=mock_rerank_response)

        mock_cohere_module = MagicMock()
        mock_cohere_module.Client = MagicMock(return_value=mock_cohere_client)
        with patch.dict(sys.modules, {"cohere": mock_cohere_module}), \
             patch("os.getenv", return_value="test-api-key"):
            reranked = await reranker.rerank("test query", results)

        assert len(reranked) == 2
        # B should be first (index 1, score 0.95)
        assert reranked[0][0].content == "Document B"
        assert reranked[0][1] == 0.95
        # A should be second (index 0, score 0.85)
        assert reranked[1][0].content == "Document A"
        assert reranked[1][1] == 0.85

    @pytest.mark.asyncio
    async def test_rerank_cohere_no_llm_client(self):
        """Cohere falls back when no llm_client provided."""
        reranker = Reranker(model="cohere", llm_client=None)
        chunks = [_make_chunk("A"), _make_chunk("B")]
        results = [(chunks[0], 0.9), (chunks[1], 0.7)]

        reranked = await reranker.rerank("test query", results)

        # Should return original results
        assert reranked == results

    @pytest.mark.asyncio
    async def test_rerank_cohere_no_api_key(self):
        """Cohere falls back when API key not set."""
        mock_llm = MagicMock()
        reranker = Reranker(llm_client=mock_llm, model="cohere")
        chunks = [_make_chunk("A"), _make_chunk("B")]
        results = [(chunks[0], 0.9), (chunks[1], 0.7)]

        with patch("os.getenv", return_value=None):
            reranked = await reranker.rerank("test query", results)

        # Should return original results
        assert reranked == results

    @pytest.mark.asyncio
    async def test_rerank_cohere_missing_dependency(self):
        """Cohere falls back when cohere package not installed."""
        mock_llm = MagicMock()
        reranker = Reranker(llm_client=mock_llm, model="cohere")
        chunks = [_make_chunk("A"), _make_chunk("B")]
        results = [(chunks[0], 0.9), (chunks[1], 0.7)]

        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "cohere":
                raise ImportError("No module")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            reranked = await reranker.rerank("test query", results)

        # Should return original results
        assert reranked == results

    @pytest.mark.asyncio
    async def test_rerank_cohere_custom_model(self):
        """Cohere uses custom model from kwargs."""
        mock_llm = MagicMock()
        reranker = Reranker(llm_client=mock_llm, model="cohere")
        reranker._kwargs["model"] = "rerank-custom"
        chunks = [_make_chunk("A")]
        results = [(chunks[0], 0.9)]

        mock_result = MagicMock()
        mock_result.index = 0
        mock_result.relevance_score = 0.8
        mock_rerank_response = MagicMock()
        mock_rerank_response.results = [mock_result]
        mock_cohere_client = MagicMock()
        mock_cohere_client.rerank = MagicMock(return_value=mock_rerank_response)

        mock_cohere_module = MagicMock()
        mock_cohere_module.Client = MagicMock(return_value=mock_cohere_client)
        with patch.dict(sys.modules, {"cohere": mock_cohere_module}), \
             patch("os.getenv", return_value="test-api-key"):
            await reranker.rerank("test query", results)

        # Check that rerank was called with custom model
        call_kwargs = mock_cohere_client.rerank.call_args[1]
        assert call_kwargs.get("model") == "rerank-custom"


class TestRerankerUnknownModel:
    """Tests for unknown reranking models."""

    @pytest.mark.asyncio
    async def test_rerank_unknown_model(self):
        """Unknown model logs warning and returns original results."""
        reranker = Reranker(model="unknown-model")
        chunks = [_make_chunk("A"), _make_chunk("B")]
        results = [(chunks[0], 0.9), (chunks[1], 0.7)]

        reranked = await reranker.rerank("test query", results)

        # Should return original results unchanged
        assert reranked == results

    @pytest.mark.asyncio
    async def test_rerank_unknown_model_with_top_k(self):
        """Unknown model applies top_k."""
        reranker = Reranker(model="unknown-model")
        chunks = [_make_chunk("A"), _make_chunk("B"), _make_chunk("C")]
        results = [(chunks[0], 0.9), (chunks[1], 0.7), (chunks[2], 0.5)]

        reranked = await reranker.rerank("test query", results, top_k=2)

        assert len(reranked) == 2
        assert reranked == results[:2]

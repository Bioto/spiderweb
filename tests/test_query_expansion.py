"""Tests for query expansion functionality.

This module tests the query expansion feature including multi-query
reformulation, HyDE strategy, and Reciprocal Rank Fusion.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from spiderweb.models.config import QueryExpansionConfig
from spiderweb.models.document import Chunk, ChunkMetadata
from spiderweb.models.result import QueryResult
from spiderweb.pipeline.query_expansion import (
    QueryExpander,
    reciprocal_rank_fusion,
    DEFAULT_MULTI_QUERY_PROMPT,
    DEFAULT_HYDE_PROMPT,
)


# Unit Tests


def test_query_expansion_config_defaults():
    """Verify default config values are sensible."""
    config = QueryExpansionConfig()
    
    assert config.enabled is False  # Opt-in feature
    assert config.strategy == "multi_query"
    assert config.custom_prompt is None
    assert config.num_expansions == 3
    assert config.include_original is True
    assert config.rrf_k == 60


def test_query_expansion_config_validation():
    """Ensure invalid configs raise appropriate errors."""
    # Valid config
    config = QueryExpansionConfig(
        enabled=True,
        num_expansions=5,
    )
    assert config.num_expansions == 5
    
    # Test boundary values
    with pytest.raises(ValueError):
        QueryExpansionConfig(num_expansions=0)  # Too low
    
    with pytest.raises(ValueError):
        QueryExpansionConfig(num_expansions=11)  # Too high
    
    with pytest.raises(ValueError):
        QueryExpansionConfig(rrf_k=0)  # Too low


def test_reciprocal_rank_fusion_basic():
    """RRF correctly combines two result sets."""
    # Create mock chunks
    chunk1 = MagicMock(spec=Chunk)
    chunk1.id = "chunk_1"
    
    chunk2 = MagicMock(spec=Chunk)
    chunk2.id = "chunk_2"
    
    chunk3 = MagicMock(spec=Chunk)
    chunk3.id = "chunk_3"
    
    # Two query results with different rankings
    query1_results = [
        (chunk1, 0.9),  # rank 1
        (chunk2, 0.7),  # rank 2
        (chunk3, 0.5),  # rank 3
    ]
    
    query2_results = [
        (chunk2, 0.95),  # rank 1
        (chunk3, 0.8),   # rank 2
        (chunk1, 0.6),   # rank 3
    ]
    
    # Apply RRF
    combined = reciprocal_rank_fusion([query1_results, query2_results], k=60)
    
    # Verify results
    assert len(combined) == 3
    
    # chunk2 should be first (rank 2 + rank 1 = best combined score)
    assert combined[0][0].id == "chunk_2"
    
    # Check that scores are RRF scores
    chunk2_score = combined[0][1]
    expected_chunk2_rrf = (1 / (60 + 2)) + (1 / (60 + 1))  # From both queries
    assert abs(chunk2_score - expected_chunk2_rrf) < 0.0001


def test_reciprocal_rank_fusion_overlapping():
    """RRF handles same document appearing in multiple queries."""
    chunk1 = MagicMock(spec=Chunk)
    chunk1.id = "chunk_1"
    
    chunk2 = MagicMock(spec=Chunk)
    chunk2.id = "chunk_2"
    
    # Chunk1 appears in all queries at rank 1
    query1_results = [(chunk1, 0.95), (chunk2, 0.5)]
    query2_results = [(chunk1, 0.90)]
    query3_results = [(chunk1, 0.92), (chunk2, 0.6)]
    
    combined = reciprocal_rank_fusion([query1_results, query2_results, query3_results], k=60)
    
    # chunk1 should have highest RRF score
    assert combined[0][0].id == "chunk_1"
    
    # Verify RRF score accumulation
    expected_rrf = 3 * (1 / (60 + 1))  # rank 1 in all 3 queries
    assert abs(combined[0][1] - expected_rrf) < 0.0001


def test_reciprocal_rank_fusion_empty():
    """RRF handles empty result sets gracefully."""
    # Empty input
    result = reciprocal_rank_fusion([])
    assert result == []
    
    # Empty sublists
    result = reciprocal_rank_fusion([[], []])
    assert result == []
    
    # Mix of empty and non-empty
    chunk1 = MagicMock(spec=Chunk)
    chunk1.id = "chunk_1"
    
    result = reciprocal_rank_fusion([[], [(chunk1, 0.9)], []])
    assert len(result) == 1
    assert result[0][0].id == "chunk_1"


def test_multi_query_prompt_formatting():
    """Default prompt correctly formats with num_expansions."""
    formatted = DEFAULT_MULTI_QUERY_PROMPT.format(
        num_expansions=3,
        query="What is Python?"
    )
    
    assert "3" in formatted
    assert "What is Python?" in formatted
    assert "alternative phrasings" in formatted.lower()


def test_hyde_prompt_formatting():
    """HyDE prompt includes query correctly."""
    formatted = DEFAULT_HYDE_PROMPT.format(query="What is Python?")
    
    assert "What is Python?" in formatted
    assert "paragraph" in formatted.lower()
    assert "document" in formatted.lower()


def test_custom_prompt_override():
    """Custom prompts are used when provided."""
    config = QueryExpansionConfig(
        enabled=True,
        custom_prompt="Custom prompt for {query}"
    )
    
    assert config.custom_prompt == "Custom prompt for {query}"


# Integration Tests (with mocked LLM)


@pytest.mark.asyncio
async def test_query_with_multi_query_expansion():
    """Full query flow with multi-query strategy."""
    # Mock LLM client
    mock_llm = AsyncMock()
    
    # Mock generate response for query expansion
    mock_generate_response = MagicMock()
    mock_generate_response.content = """What is machine learning?
Define machine learning
Explain ML concepts"""
    mock_llm.generate = AsyncMock(return_value=mock_generate_response)
    
    # Mock embedding responses
    mock_embed_response = MagicMock()
    mock_embed_response.embeddings = [[0.1] * 1536]
    mock_llm.embed = AsyncMock(return_value=mock_embed_response)
    
    # Create expander
    config = QueryExpansionConfig(
        enabled=True,
        strategy="multi_query",
        num_expansions=3,
        include_original=True,
    )
    expander = QueryExpander(mock_llm, config)
    
    # Expand query
    expanded = await expander.expand("What is machine learning?")
    
    # Should include original + 3 expansions
    assert len(expanded) == 4
    assert expanded[0] == "What is machine learning?"
    assert "machine learning" in expanded[1].lower() or "ml" in expanded[1].lower()


@pytest.mark.asyncio
async def test_query_with_hyde_expansion():
    """Full query flow with HyDE strategy."""
    # Mock LLM client
    mock_llm = AsyncMock()
    
    # Mock generate response for HyDE
    mock_generate_response = MagicMock()
    mock_generate_response.content = """Machine learning is a branch of artificial intelligence that enables computers to learn from data without explicit programming. It uses statistical techniques to give computer systems the ability to progressively improve performance on a specific task. Common applications include image recognition, natural language processing, and predictive analytics."""
    mock_llm.generate = AsyncMock(return_value=mock_generate_response)
    
    # Create expander
    config = QueryExpansionConfig(
        enabled=True,
        strategy="hyde",
        num_expansions=1,
        include_original=False,
    )
    expander = QueryExpander(mock_llm, config)
    
    # Expand query
    expanded = await expander.expand("What is machine learning?")
    
    # Should only have the hypothetical document (no original)
    assert len(expanded) == 1
    assert "artificial intelligence" in expanded[0].lower()


@pytest.mark.asyncio
async def test_query_expansion_disabled_by_default():
    """Expansion doesn't run unless explicitly enabled."""
    config = QueryExpansionConfig()  # defaults to enabled=False
    
    assert config.enabled is False


@pytest.mark.asyncio
async def test_query_expansion_includes_original():
    """Original query is included when include_original=True."""
    mock_llm = AsyncMock()
    
    mock_generate_response = MagicMock()
    mock_generate_response.content = "Alternative query"
    mock_llm.generate = AsyncMock(return_value=mock_generate_response)
    
    config = QueryExpansionConfig(
        enabled=True,
        num_expansions=1,
        include_original=True,
    )
    expander = QueryExpander(mock_llm, config)
    
    expanded = await expander.expand("Original query")
    
    assert len(expanded) == 2
    assert expanded[0] == "Original query"


@pytest.mark.asyncio
async def test_query_expansion_excludes_original():
    """Original query excluded when include_original=False."""
    mock_llm = AsyncMock()
    
    mock_generate_response = MagicMock()
    mock_generate_response.content = "Alternative query"
    mock_llm.generate = AsyncMock(return_value=mock_generate_response)
    
    config = QueryExpansionConfig(
        enabled=True,
        num_expansions=1,
        include_original=False,
    )
    expander = QueryExpander(mock_llm, config)
    
    expanded = await expander.expand("Original query")
    
    assert len(expanded) == 1
    assert expanded[0] != "Original query"


@pytest.mark.asyncio
async def test_query_result_contains_expansion_metadata():
    """Result includes expanded_queries and expansion_strategy."""
    # Create a result with expansion metadata
    result = QueryResult(
        query="What is Python?",
        chunks=[],
        scores=[],
        execution_time_seconds=0.5,
        total_results=0,
        expanded_queries=["What is Python?", "Define Python", "Python explanation"],
        expansion_strategy="multi_query",
        rrf_scores=[0.03, 0.025, 0.02],
    )
    
    assert result.expanded_queries is not None
    assert len(result.expanded_queries) == 3
    assert result.expansion_strategy == "multi_query"
    assert result.rrf_scores is not None
    assert len(result.rrf_scores) == 3


@pytest.mark.asyncio
async def test_query_expansion_with_context_window():
    """Expansion works correctly combined with context retrieval."""
    # This test verifies that both features can be enabled simultaneously
    from spiderweb.models.config import ContextWindowConfig
    
    expansion_config = QueryExpansionConfig(enabled=True)
    context_config = ContextWindowConfig(enabled=True, chunks_before=2, chunks_after=2)
    
    # Both configs should be independent
    assert expansion_config.enabled
    assert context_config.enabled


# Edge Cases


@pytest.mark.asyncio
async def test_expansion_with_empty_query():
    """Handles empty/whitespace queries."""
    mock_llm = AsyncMock()
    config = QueryExpansionConfig(enabled=True)
    expander = QueryExpander(mock_llm, config)
    
    # Empty query should raise ValueError
    with pytest.raises(ValueError, match="Query cannot be empty"):
        await expander.expand("")
    
    with pytest.raises(ValueError, match="Query cannot be empty"):
        await expander.expand("   ")


@pytest.mark.asyncio
async def test_expansion_llm_error_fallback():
    """Falls back to original query if LLM expansion fails."""
    mock_llm = AsyncMock()
    
    # Simulate LLM error
    mock_llm.generate = AsyncMock(side_effect=Exception("LLM API Error"))
    
    config = QueryExpansionConfig(enabled=True)
    expander = QueryExpander(mock_llm, config)
    
    # Should fall back to original query
    expanded = await expander.expand("Test query")
    
    assert len(expanded) == 1
    assert expanded[0] == "Test query"


@pytest.mark.asyncio
async def test_expansion_with_special_characters():
    """Queries with quotes, unicode, etc. handled correctly."""
    mock_llm = AsyncMock()
    
    mock_generate_response = MagicMock()
    mock_generate_response.content = "Expanded query"
    mock_llm.generate = AsyncMock(return_value=mock_generate_response)
    
    config = QueryExpansionConfig(enabled=True, num_expansions=1, include_original=True)
    expander = QueryExpander(mock_llm, config)
    
    # Query with various special characters
    special_query = 'What is "machine learning" in français? 🤖'
    expanded = await expander.expand(special_query)
    
    # Should handle special characters
    assert len(expanded) == 2
    assert expanded[0] == special_query


@pytest.mark.asyncio
async def test_multi_query_with_numbered_output():
    """Handles LLM output with numbering/bullets."""
    mock_llm = AsyncMock()
    
    # LLM returns numbered list
    mock_generate_response = MagicMock()
    mock_generate_response.content = """1. What is machine learning?
2. Define machine learning
3. Explain ML concepts"""
    mock_llm.generate = AsyncMock(return_value=mock_generate_response)
    
    config = QueryExpansionConfig(
        enabled=True,
        strategy="multi_query",
        num_expansions=3,
        include_original=False,
    )
    expander = QueryExpander(mock_llm, config)
    
    expanded = await expander.expand("ML query")
    
    # Should strip numbering
    assert len(expanded) == 3
    assert not expanded[0].startswith("1")
    assert "machine learning" in expanded[0].lower()


@pytest.mark.asyncio
async def test_multi_query_with_bullet_points():
    """Handles LLM output with bullet points."""
    mock_llm = AsyncMock()
    
    # LLM returns bulleted list
    mock_generate_response = MagicMock()
    mock_generate_response.content = """- What is machine learning?
* Define machine learning
• Explain ML concepts"""
    mock_llm.generate = AsyncMock(return_value=mock_generate_response)
    
    config = QueryExpansionConfig(
        enabled=True,
        strategy="multi_query",
        num_expansions=3,
        include_original=False,
    )
    expander = QueryExpander(mock_llm, config)
    
    expanded = await expander.expand("ML query")
    
    # Should strip bullet points
    assert len(expanded) == 3
    for query in expanded:
        assert not query.startswith("-")
        assert not query.startswith("*")
        assert not query.startswith("•")


@pytest.mark.asyncio
async def test_rrf_k_parameter_effect():
    """Test that different RRF k values affect scoring."""
    chunk1 = MagicMock(spec=Chunk)
    chunk1.id = "chunk_1"
    
    results = [[(chunk1, 0.9)]]
    
    # Lower k means higher scores
    rrf_low_k = reciprocal_rank_fusion(results, k=10)
    rrf_high_k = reciprocal_rank_fusion(results, k=100)
    
    # With k=10: 1/(10+1) = 0.0909
    # With k=100: 1/(100+1) = 0.0099
    assert rrf_low_k[0][1] > rrf_high_k[0][1]


@pytest.mark.asyncio
async def test_hyde_multiple_expansions():
    """HyDE can generate multiple hypothetical documents."""
    mock_llm = AsyncMock()
    
    # Create side effects for multiple generate calls
    responses = [
        MagicMock(content="First hypothetical document about ML."),
        MagicMock(content="Second hypothetical document with different wording."),
        MagicMock(content="Third variation of hypothetical content."),
    ]
    mock_llm.generate = AsyncMock(side_effect=responses)
    
    config = QueryExpansionConfig(
        enabled=True,
        strategy="hyde",
        num_expansions=3,
        include_original=False,
    )
    expander = QueryExpander(mock_llm, config)
    
    expanded = await expander.expand("What is ML?")
    
    # Should have 3 different hypothetical documents
    assert len(expanded) == 3
    assert expanded[0] != expanded[1]
    assert expanded[1] != expanded[2]


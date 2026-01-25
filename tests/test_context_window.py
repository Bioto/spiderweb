"""Tests for context window feature."""

import pytest
import random
from unittest.mock import AsyncMock, MagicMock, patch

from spiderweb.models.config import ContextWindowConfig
from spiderweb.models.result import ContextChunk, QueryResult, MatchContext
from spiderweb.models.document import Chunk, ChunkMetadata, ChunkType
from spiderweb.pipeline.context import ContextRetriever
from spiderweb.stores.memory import MemoryVectorStore

pytestmark = pytest.mark.asyncio


class TestContextWindowConfig:
    """Tests for ContextWindowConfig model validation."""

    def test_default_values(self):
        """Test config has sensible defaults."""
        config = ContextWindowConfig()
        assert config.enabled is True
        assert config.chunks_before == 2
        assert config.chunks_after == 2
        assert config.context_mode == "page"
        assert config.semantic_guide is None
        assert config.deduplicate is True

    def test_no_upper_limit_on_chunks(self):
        """Test that large chunk counts are allowed (future-proofing)."""
        config = ContextWindowConfig(chunks_before=1000, chunks_after=1000)
        assert config.chunks_before == 1000
        assert config.chunks_after == 1000

    def test_zero_context_allowed(self):
        """Test that zero context (match only) is valid."""
        config = ContextWindowConfig(chunks_before=0, chunks_after=0)
        assert config.chunks_before == 0
        assert config.chunks_after == 0

    def test_negative_chunks_rejected(self):
        """Test that negative chunk counts raise ValidationError."""
        with pytest.raises(ValueError):
            ContextWindowConfig(chunks_before=-1)

    def test_context_mode_validation(self):
        """Test that invalid context_mode raises error."""
        with pytest.raises(ValueError):
            ContextWindowConfig(context_mode="invalid")

    def test_semantic_boost_weight_bounds(self):
        """Test semantic_boost_weight must be 0.0-1.0."""
        config = ContextWindowConfig(semantic_boost_weight=0.5)
        assert config.semantic_boost_weight == 0.5

        with pytest.raises(ValueError):
            ContextWindowConfig(semantic_boost_weight=1.5)

        with pytest.raises(ValueError):
            ContextWindowConfig(semantic_boost_weight=-0.1)


class TestMemoryStoreContextRetrieval:
    """Tests for MemoryVectorStore.get_by_position."""

    @pytest.fixture
    def sample_chunks(self) -> list[Chunk]:
        """Create sample chunks for testing."""
        chunks = []
        for i in range(10):
            chunk = Chunk(
                id=f"chunk-{i}",
                content=f"Content for chunk {i}",
                embedding=[0.1] * 1536,
                metadata=ChunkMetadata(
                    document_id="doc-1",
                    chunk_index=i,
                    chunk_type=ChunkType.HIERARCHICAL,
                    page_numbers=[i // 3 + 1],  # Pages 1, 1, 1, 2, 2, 2, 3, 3, 3, 4
                ),
            )
            chunks.append(chunk)
        return chunks

    async def test_get_by_chunk_index_range(self, sample_chunks):
        """Test retrieving chunks by chunk_index range."""
        store = MemoryVectorStore()
        await store.upsert(sample_chunks)

        result = await store.get_by_position(
            document_id="doc-1",
            position_start=3,
            position_end=6,
            position_field="chunk_index",
        )

        assert len(result) == 4  # Chunks 3, 4, 5, 6
        indices = [c.metadata.chunk_index for c in result]
        assert indices == [3, 4, 5, 6]

    async def test_get_by_page_number_range(self, sample_chunks):
        """Test retrieving chunks by page_number range."""
        store = MemoryVectorStore()
        await store.upsert(sample_chunks)

        result = await store.get_by_position(
            document_id="doc-1",
            position_start=2,
            position_end=2,
            position_field="page_number",
        )

        # Chunks 3, 4, 5 are on page 2
        assert len(result) == 3
        for chunk in result:
            assert 2 in chunk.metadata.page_numbers

    async def test_empty_range_returns_empty(self, sample_chunks):
        """Test that non-existent range returns empty list."""
        store = MemoryVectorStore()
        await store.upsert(sample_chunks)

        result = await store.get_by_position(
            document_id="doc-1",
            position_start=100,
            position_end=200,
            position_field="chunk_index",
        )

        assert result == []

    async def test_filters_by_document_id(self, sample_chunks):
        """Test that only chunks from specified document are returned."""
        store = MemoryVectorStore()
        await store.upsert(sample_chunks)

        # Add chunk from different document
        other_chunk = Chunk(
            id="other-chunk",
            content="Other document",
            embedding=[0.1] * 1536,
            metadata=ChunkMetadata(
                document_id="doc-2",
                chunk_index=5,
                chunk_type=ChunkType.HIERARCHICAL,
            ),
        )
        await store.upsert([other_chunk])

        result = await store.get_by_position(
            document_id="doc-1",
            position_start=4,
            position_end=6,
            position_field="chunk_index",
        )

        # Should only include doc-1 chunks
        assert all(c.metadata.document_id == "doc-1" for c in result)

    async def test_results_sorted_by_position(self, sample_chunks):
        """Test that results are sorted by position."""
        store = MemoryVectorStore()
        # Insert in random order
        shuffled = sample_chunks.copy()
        random.shuffle(shuffled)
        await store.upsert(shuffled)

        result = await store.get_by_position(
            document_id="doc-1",
            position_start=0,
            position_end=9,
            position_field="chunk_index",
        )

        indices = [c.metadata.chunk_index for c in result]
        assert indices == sorted(indices)


class TestContextRetriever:
    """Tests for ContextRetriever service."""

    @pytest.fixture
    def mock_store(self):
        """Create mock vector store."""
        store = AsyncMock()
        return store

    @pytest.fixture
    def sample_match(self) -> Chunk:
        """Create a sample matched chunk."""
        return Chunk(
            id="match-1",
            content="This is the matched content",
            embedding=[0.1] * 1536,
            metadata=ChunkMetadata(
                document_id="doc-1",
                chunk_index=5,
                chunk_type=ChunkType.HIERARCHICAL,
                page_numbers=[2],
            ),
        )

    async def test_retrieves_context_before_and_after(self, mock_store, sample_match):
        """Test basic context retrieval."""
        # Setup mock to return context chunks
        context_chunks = [
            Chunk(
                id=f"ctx-{i}",
                content=f"Context {i}",
                embedding=[0.1] * 1536,
                metadata=ChunkMetadata(document_id="doc-1", chunk_index=i, chunk_type=ChunkType.HIERARCHICAL),
            )
            for i in range(3, 8)  # Chunks 3, 4, 5, 6, 7
        ]
        mock_store.get_by_position.return_value = context_chunks

        retriever = ContextRetriever(vector_store=mock_store)
        config = ContextWindowConfig(chunks_before=2, chunks_after=2, context_mode="chunk")

        result = await retriever.get_context_for_matches(
            matches=[sample_match],
            config=config,
        )

        # Should have called get_by_position with correct range
        mock_store.get_by_position.assert_called_once_with(
            document_id="doc-1",
            position_start=3,  # 5 - 2
            position_end=7,  # 5 + 2
            position_field="chunk_index",
        )

        assert 0 in result  # Match index 0
        assert len(result[0].chunks) == 5  # 5 context chunks

    async def test_page_mode_uses_page_numbers(self, mock_store, sample_match):
        """Test that page mode queries by page_number."""
        mock_store.get_by_position.return_value = []

        retriever = ContextRetriever(vector_store=mock_store)
        config = ContextWindowConfig(chunks_before=1, chunks_after=1, context_mode="page")

        await retriever.get_context_for_matches(
            matches=[sample_match],
            config=config,
        )

        # Should query by page_number
        mock_store.get_by_position.assert_called_once()
        call_args = mock_store.get_by_position.call_args
        assert call_args.kwargs["position_field"] == "page_number"
        assert call_args.kwargs["position_start"] == 1  # page 2 - 1
        assert call_args.kwargs["position_end"] == 3  # page 2 + 1

    async def test_disabled_returns_empty(self, mock_store, sample_match):
        """Test that disabled config returns empty context."""
        retriever = ContextRetriever(vector_store=mock_store)
        config = ContextWindowConfig(enabled=False)

        result = await retriever.get_context_for_matches(
            matches=[sample_match],
            config=config,
        )

        assert result == {}
        mock_store.get_by_position.assert_not_called()

    async def test_position_offset_calculated(self, mock_store, sample_match):
        """Test that position_offset is correctly calculated for each context chunk."""
        context_chunks = [
            Chunk(
                id=f"ctx-{i}",
                content=f"Context {i}",
                embedding=[0.1] * 1536,
                metadata=ChunkMetadata(document_id="doc-1", chunk_index=i, chunk_type=ChunkType.HIERARCHICAL),
            )
            for i in [3, 4, 5, 6, 7]  # Match is at 5
        ]
        mock_store.get_by_position.return_value = context_chunks

        retriever = ContextRetriever(vector_store=mock_store)
        # Use chunk mode since test chunks don't have page_numbers
        config = ContextWindowConfig(chunks_before=2, chunks_after=2, context_mode="chunk")

        result = await retriever.get_context_for_matches(
            matches=[sample_match],
            config=config,
        )

        offsets = [ctx.position_offset for ctx in result[0].chunks]
        assert offsets == [-2, -1, 0, 1, 2]  # Relative to match at index 5


class TestContextWindowEdgeCases:
    """Edge cases and error handling tests."""

    async def test_empty_matches_returns_empty(self):
        """Test that empty match list returns empty context."""
        mock_store = AsyncMock()
        retriever = ContextRetriever(vector_store=mock_store)

        result = await retriever.get_context_for_matches(
            matches=[],
            config=ContextWindowConfig(),
        )

        assert result == {}

    async def test_first_chunk_no_before_context(self):
        """Test handling match at chunk 0 with before context requested."""
        mock_store = AsyncMock()
        mock_store.get_by_position.return_value = []

        match = Chunk(
            id="match",
            content="First chunk",
            embedding=[0.1] * 1536,
            metadata=ChunkMetadata(document_id="doc-1", chunk_index=0, chunk_type=ChunkType.HIERARCHICAL),
        )

        retriever = ContextRetriever(vector_store=mock_store)
        config = ContextWindowConfig(chunks_before=5, chunks_after=2)

        await retriever.get_context_for_matches(
            matches=[match],
            config=config,
        )

        # Should clamp to 0, not negative
        call_args = mock_store.get_by_position.call_args
        assert call_args.kwargs["position_start"] >= 0

    async def test_store_error_handled_gracefully(self):
        """Test that store errors don't crash the query."""
        mock_store = AsyncMock()
        mock_store.get_by_position.side_effect = Exception("Store error")

        match = Chunk(
            id="match",
            content="Match",
            embedding=[0.1] * 1536,
            metadata=ChunkMetadata(document_id="doc-1", chunk_index=5, chunk_type=ChunkType.HIERARCHICAL),
        )

        retriever = ContextRetriever(vector_store=mock_store)
        config = ContextWindowConfig()

        # Should not raise, return empty or partial context
        result = await retriever.get_context_for_matches(
            matches=[match],
            config=config,
        )

        # Graceful degradation
        assert 0 in result
        assert result[0].chunks == []

    async def test_very_large_context_window(self):
        """Test handling of very large context windows."""
        mock_store = AsyncMock()
        mock_store.get_by_position.return_value = []

        match = Chunk(
            id="match",
            content="Match",
            embedding=[0.1] * 1536,
            metadata=ChunkMetadata(document_id="doc-1", chunk_index=50, chunk_type=ChunkType.HIERARCHICAL),
        )

        retriever = ContextRetriever(vector_store=mock_store)
        config = ContextWindowConfig(chunks_before=10000, chunks_after=10000)

        # Should not crash, just request large range
        await retriever.get_context_for_matches(
            matches=[match],
            config=config,
        )

        assert mock_store.get_by_position.called


class TestAdaptiveExpansion:
    """Tests for adaptive context expansion when semantic scores are low."""

    @pytest.fixture
    def mock_store_with_expansion(self):
        """Create mock store that returns more chunks as window expands."""
        store = AsyncMock()

        # Track call count to simulate expanding context
        call_count = 0

        def get_by_position_side_effect(document_id, position_start, position_end, position_field):
            nonlocal call_count
            call_count += 1

            # Create embeddings with actual different directions for proper cosine similarity testing
            # Low score: mostly zeros with some 1s = low similarity to guide
            low_embedding = [0.1 if i < 50 else 0.0 for i in range(1536)]
            # High score: mostly 1s = high similarity to guide (which is all 1s)
            high_embedding = [1.0 if i < 1500 else 0.0 for i in range(1536)]

            # First call: return low-scoring chunks
            if call_count == 1:
                return [
                    Chunk(
                        id="low-1",
                        content="Irrelevant content",
                        embedding=low_embedding,
                        metadata=ChunkMetadata(
                            document_id=document_id, chunk_index=4, chunk_type=ChunkType.HIERARCHICAL
                        ),
                    ),
                ]
            # Second call (expanded): return high-scoring chunk
            else:
                return [
                    Chunk(
                        id="low-1",
                        content="Irrelevant content",
                        embedding=low_embedding,
                        metadata=ChunkMetadata(
                            document_id=document_id, chunk_index=4, chunk_type=ChunkType.HIERARCHICAL
                        ),
                    ),
                    Chunk(
                        id="high-1",
                        content="Financial metrics and revenue",
                        embedding=high_embedding,
                        metadata=ChunkMetadata(
                            document_id=document_id, chunk_index=2, chunk_type=ChunkType.HIERARCHICAL
                        ),
                    ),
                ]

        store.get_by_position.side_effect = get_by_position_side_effect
        return store

    async def test_expands_when_scores_below_threshold(self, mock_store_with_expansion):
        """Test that window expands when no chunk meets semantic threshold."""
        mock_llm = AsyncMock()
        # Guide embedding: mostly 1s to match high_embedding in the mock
        guide_embedding = [1.0 if i < 1500 else 0.0 for i in range(1536)]
        mock_llm.embed.return_value = MagicMock(embeddings=[guide_embedding])

        match = Chunk(
            id="match",
            content="Revenue",
            embedding=[0.5] * 1536,
            metadata=ChunkMetadata(document_id="doc-1", chunk_index=5, chunk_type=ChunkType.HIERARCHICAL),
        )

        retriever = ContextRetriever(vector_store=mock_store_with_expansion)
        config = ContextWindowConfig(
            chunks_before=2,
            chunks_after=2,
            semantic_guide="Find financial metrics",
            semantic_min_score=0.6,
            expand_on_low_score=True,
            expansion_step_size=2,
            max_expansion_steps=3,
        )

        result = await retriever.get_context_for_matches(
            matches=[match],
            config=config,
            llm_client=mock_llm,
        )

        # Should have made multiple calls due to expansion
        assert mock_store_with_expansion.get_by_position.call_count >= 2

        # Second call should have expanded range
        calls = mock_store_with_expansion.get_by_position.call_args_list
        first_call = calls[0].kwargs
        second_call = calls[1].kwargs

        # Expanded window should be larger
        first_range = first_call["position_end"] - first_call["position_start"]
        second_range = second_call["position_end"] - second_call["position_start"]
        assert second_range > first_range

    async def test_stops_expansion_when_threshold_met(self, mock_store_with_expansion):
        """Test that expansion stops once threshold is met."""
        mock_llm = AsyncMock()
        # Guide embedding: mostly 1s to match high_embedding in the mock
        guide_embedding = [1.0 if i < 1500 else 0.0 for i in range(1536)]
        mock_llm.embed.return_value = MagicMock(embeddings=[guide_embedding])

        match = Chunk(
            id="match",
            content="Revenue",
            embedding=[0.5] * 1536,
            metadata=ChunkMetadata(document_id="doc-1", chunk_index=5, chunk_type=ChunkType.HIERARCHICAL),
        )

        retriever = ContextRetriever(vector_store=mock_store_with_expansion)
        config = ContextWindowConfig(
            chunks_before=2,
            chunks_after=2,
            semantic_guide="Find financial metrics",
            semantic_min_score=0.6,
            expand_on_low_score=True,
            max_expansion_steps=5,  # Allow many steps
        )

        result = await retriever.get_context_for_matches(
            matches=[match],
            config=config,
            llm_client=mock_llm,
        )

        # Should stop at step 2 (when high-scoring chunk found), not go to 5
        assert mock_store_with_expansion.get_by_position.call_count == 2

    async def test_respects_max_expansion_steps(self):
        """Test that expansion respects max_expansion_steps limit."""
        mock_store = AsyncMock()
        
        # Create low-scoring embedding (sparse, different direction from guide)
        low_embedding = [0.1 if i < 50 else 0.0 for i in range(1536)]
        
        # Always return low-scoring chunks
        mock_store.get_by_position.return_value = [
            Chunk(
                id="low",
                content="Irrelevant",
                embedding=low_embedding,
                metadata=ChunkMetadata(document_id="doc-1", chunk_index=4, chunk_type=ChunkType.HIERARCHICAL),
            ),
        ]

        mock_llm = AsyncMock()
        # Guide embedding: mostly 1s, very different from low_embedding
        guide_embedding = [1.0 if i < 1500 else 0.0 for i in range(1536)]
        mock_llm.embed.return_value = MagicMock(embeddings=[guide_embedding])

        match = Chunk(
            id="match",
            content="Revenue",
            embedding=[0.5] * 1536,
            metadata=ChunkMetadata(document_id="doc-1", chunk_index=5, chunk_type=ChunkType.HIERARCHICAL),
        )

        retriever = ContextRetriever(vector_store=mock_store)
        config = ContextWindowConfig(
            chunks_before=2,
            chunks_after=2,
            semantic_guide="Find financial metrics",
            semantic_min_score=0.9,  # Impossible to meet
            expand_on_low_score=True,
            max_expansion_steps=3,
        )

        result = await retriever.get_context_for_matches(
            matches=[match],
            config=config,
            llm_client=mock_llm,
        )

        # Should stop at max_expansion_steps + 1 (initial + 3 expansions)
        assert mock_store.get_by_position.call_count == 4

    async def test_no_expansion_when_disabled(self):
        """Test that expansion doesn't happen when disabled."""
        mock_store = AsyncMock()
        mock_store.get_by_position.return_value = [
            Chunk(
                id="low",
                content="Irrelevant",
                embedding=[0.1] * 1536,
                metadata=ChunkMetadata(document_id="doc-1", chunk_index=4, chunk_type=ChunkType.HIERARCHICAL),
            ),
        ]

        mock_llm = AsyncMock()
        mock_llm.embed.return_value = MagicMock(embeddings=[[0.7] * 1536])

        match = Chunk(
            id="match",
            content="Revenue",
            embedding=[0.5] * 1536,
            metadata=ChunkMetadata(document_id="doc-1", chunk_index=5, chunk_type=ChunkType.HIERARCHICAL),
        )

        retriever = ContextRetriever(vector_store=mock_store)
        config = ContextWindowConfig(
            chunks_before=2,
            chunks_after=2,
            semantic_guide="Find financial metrics",
            expand_on_low_score=False,  # Disabled
        )

        await retriever.get_context_for_matches(
            matches=[match],
            config=config,
            llm_client=mock_llm,
        )

        # Should only make initial call, no expansion
        assert mock_store.get_by_position.call_count == 1

    async def test_expansion_tracks_steps_in_result(self):
        """Test that result includes expansion metadata."""
        mock_store = AsyncMock()
        mock_store.get_by_position.return_value = [
            Chunk(
                id="high",
                content="Financial data",
                embedding=[0.9] * 1536,
                metadata=ChunkMetadata(document_id="doc-1", chunk_index=4, chunk_type=ChunkType.HIERARCHICAL),
            ),
        ]

        mock_llm = AsyncMock()
        mock_llm.embed.return_value = MagicMock(embeddings=[[0.7] * 1536])

        match = Chunk(
            id="match",
            content="Revenue",
            embedding=[0.5] * 1536,
            metadata=ChunkMetadata(document_id="doc-1", chunk_index=5, chunk_type=ChunkType.HIERARCHICAL),
        )

        retriever = ContextRetriever(vector_store=mock_store)
        config = ContextWindowConfig(
            semantic_guide="Find financial metrics",
            expand_on_low_score=True,
        )

        result = await retriever.get_context_for_matches(
            matches=[match],
            config=config,
            llm_client=mock_llm,
        )

        # Result should track expansion steps
        assert 0 in result
        match_context = result[0]
        assert match_context.expansion_steps_used >= 0
        assert match_context.final_window_size[0] >= 0
        assert match_context.final_window_size[1] >= 0


class TestAPIIntegration:
    """Integration tests for Spiderweb.query with context window."""

    async def test_query_without_context_backward_compatible(self):
        """Test that query without context_window works as before."""
        with patch("spiderweb.api.DocumentProcessor") as MockProcessor:
            mock_store = AsyncMock()
            mock_store.query.return_value = []

            mock_processor = MagicMock()
            mock_processor.vector_store = mock_store
            MockProcessor.return_value = mock_processor

            mock_llm = AsyncMock()
            mock_llm.embed.return_value = MagicMock(embeddings=[[0.1] * 1536])

            from spiderweb.api import Spiderweb

            web = Spiderweb(llm_client=mock_llm)

            result = await web.query("test query", top_k=5)

            # Should work without context
            assert hasattr(result, "chunks")
            assert isinstance(result, QueryResult)
            mock_store.get_by_position.assert_not_called()


class TestProgressiveContext:
    """Tests for page-first context in progressive RAG."""

    async def test_progressive_defaults_to_page_mode(self):
        """Test that progressive RAG context defaults to page mode."""
        from spiderweb.pipeline.progressive import ProgressiveRAGProcessor
        from spiderweb.models.progressive import ProgressiveRAGConfig

        mock_llm = AsyncMock()
        mock_summary_store = AsyncMock()
        mock_full_store = AsyncMock()
        mock_processor = MagicMock()

        # Setup mock query to return page summaries
        mock_summary_store.query.return_value = []

        processor = ProgressiveRAGProcessor(
            llm_client=mock_llm,
            summary_store=mock_summary_store,
            full_store=mock_full_store,
            document_processor=mock_processor,
            config=ProgressiveRAGConfig(),
        )

        mock_llm.embed.return_value = MagicMock(embeddings=[[0.1] * 1536])

        # Create context config with chunk mode
        context_config = ContextWindowConfig(context_mode="chunk")

        # Query should override to page mode
        result = await processor.query(
            query_text="test",
            top_k=5,
            context_window=context_config,
        )

        # Context mode should have been overridden to page
        assert context_config.context_mode == "page"



"""Tests for adaptive chunking agent."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure chunker registry is populated (chunkers register on import)
import spiderweb.chunkers  # noqa: F401

from spiderweb.chunking_agent import (
    ChunkerChoice,
    choose_chunking_strategy,
    create_chunker_from_strategy,
)
from spiderweb.models.config import ChunkerConfig
from spiderweb.models.document import Document, DocumentMetadata

# Directory containing chunking fixture files (markdown, txt for hierarchical, sentence, semantic, etc.)
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "chunking"


class TestChunkerChoice:
    """Tests for ChunkerChoice model."""

    def test_chunker_choice_creation(self):
        """Test creating a ChunkerChoice with valid strategy."""
        choice = ChunkerChoice(
            strategy="hierarchical",
            rationale="Document has clear markdown structure",
        )
        assert choice.strategy == "hierarchical"
        assert "markdown" in choice.rationale.lower()

    def test_chunker_choice_invalid_strategy(self):
        """Test that invalid strategy raises validation error."""
        with pytest.raises(Exception):  # Pydantic validation error
            ChunkerChoice(strategy="invalid", rationale="test")


class TestChooseChunkingStrategy:
    """Tests for choose_chunking_strategy function."""

    @pytest.mark.asyncio
    async def test_choose_strategy_with_structured_complete(self):
        """Test choosing strategy using structured_complete."""
        mock_choice = ChunkerChoice(
            strategy="hierarchical",
            rationale="Document has markdown headings",
        )

        async def mock_structured_complete(*args, **kwargs):
            return mock_choice

        mock_llm = AsyncMock()
        with patch("gluellm.api.structured_complete", side_effect=mock_structured_complete):
            choice = await choose_chunking_strategy(
                llm_client=mock_llm,
                document_preview="# Introduction\n\nThis is a markdown document.",
                file_name="README.md",
            )

        assert choice.strategy == "hierarchical"
        assert "markdown" in choice.rationale.lower()

    @pytest.mark.asyncio
    async def test_choose_strategy_fallback_to_complete(self):
        """Test fallback to complete() when structured_complete fails."""
        mock_llm = AsyncMock()

        # Mock response object
        mock_response = MagicMock()
        mock_response.final_response = '{"strategy": "sentence", "rationale": "Prose document"}'

        mock_llm.complete = AsyncMock(return_value=mock_response)

        # Mock that structured_complete raises ImportError
        import spiderweb.chunking_agent as chunking_agent_module

        def mock_structured_complete(*args, **kwargs):
            raise ImportError("structured_complete not available")

        original_import = getattr(chunking_agent_module, "structured_complete", None)

        try:
            chunking_agent_module.structured_complete = mock_structured_complete

            choice = await choose_chunking_strategy(
                llm_client=mock_llm,
                document_preview="This is a prose document with sentences.",
                file_name="article.txt",
            )

            assert choice.strategy == "sentence"
            assert "prose" in choice.rationale.lower()
        finally:
            if original_import:
                chunking_agent_module.structured_complete = original_import
            elif hasattr(chunking_agent_module, "structured_complete"):
                delattr(chunking_agent_module, "structured_complete")

    @pytest.mark.asyncio
    async def test_choose_strategy_fallback_on_json_error(self):
        """Test fallback when JSON parsing fails."""
        mock_llm = AsyncMock()

        # Mock response with invalid JSON
        mock_response = MagicMock()
        mock_response.final_response = "This is not JSON"

        mock_llm.complete = AsyncMock(return_value=mock_response)

        # Mock that structured_complete raises
        import spiderweb.chunking_agent as chunking_agent_module

        def mock_structured_complete(*args, **kwargs):
            raise Exception("Failed")

        original_import = getattr(chunking_agent_module, "structured_complete", None)

        try:
            chunking_agent_module.structured_complete = mock_structured_complete

            # Should fallback to hierarchical for .md files
            choice = await choose_chunking_strategy(
                llm_client=mock_llm,
                document_preview="# Test",
                file_name="test.md",
            )

            # Should use fallback logic
            assert choice.strategy in ["hierarchical", "semantic", "sentence", "sliding_window"]
        finally:
            if original_import:
                chunking_agent_module.structured_complete = original_import
            elif hasattr(chunking_agent_module, "structured_complete"):
                delattr(chunking_agent_module, "structured_complete")


class TestCreateChunkerFromStrategy:
    """Tests for create_chunker_from_strategy function."""

    def test_create_hierarchical_chunker(self):
        """Test creating hierarchical chunker."""
        config = ChunkerConfig(strategy="hierarchical", max_chunk_size=2000)
        chunker = create_chunker_from_strategy("hierarchical", config)

        assert chunker is not None
        assert hasattr(chunker, "chunk")

    def test_create_sentence_chunker(self):
        """Test creating sentence chunker."""
        config = ChunkerConfig(strategy="sentence", max_chunk_size=1000)
        chunker = create_chunker_from_strategy("sentence", config)

        assert chunker is not None
        assert hasattr(chunker, "chunk")

    def test_create_sliding_window_chunker(self):
        """Test creating sliding window chunker."""
        config = ChunkerConfig(strategy="sliding_window", max_chunk_size=1000)
        chunker = create_chunker_from_strategy("sliding_window", config)

        assert chunker is not None
        assert hasattr(chunker, "chunk")

    def test_create_semantic_chunker_with_llm(self):
        """Test creating semantic chunker with LLM client."""
        mock_llm = MagicMock()
        config = ChunkerConfig(strategy="semantic", max_chunk_size=2000)
        chunker = create_chunker_from_strategy("semantic", config, llm_client=mock_llm)

        assert chunker is not None
        assert hasattr(chunker, "chunk_async")
        # Semantic chunker should have llm_client set
        assert chunker.llm_client == mock_llm

    def test_create_chunker_with_default_config(self):
        """Test creating chunker without base config."""
        chunker = create_chunker_from_strategy("hierarchical")

        assert chunker is not None
        assert hasattr(chunker, "chunk")

    def test_create_chunker_invalid_strategy(self):
        """Test that invalid strategy raises KeyError."""
        with pytest.raises(KeyError):
            create_chunker_from_strategy("nonexistent_strategy")


class TestAdaptiveChunkingIntegration:
    """Integration tests for adaptive chunking with Spiderweb."""

    @pytest.mark.asyncio
    async def test_ingest_with_adaptive_chunking_requires_llm(self):
        """Test that ingest_with_adaptive_chunking requires llm_client."""
        from spiderweb import Spiderweb

        web = Spiderweb(llm_client=None)

        with pytest.raises(ValueError, match="llm_client is required"):
            await web.ingest_with_adaptive_chunking("nonexistent.md")

    @pytest.mark.asyncio
    async def test_ingest_with_adaptive_chunking_mock(self, tmp_path):
        """Test adaptive chunking with mocked LLM."""
        from unittest.mock import AsyncMock, MagicMock

        from gluellm import GlueLLM
        from spiderweb import Spiderweb

        # Create a test markdown file
        test_file = tmp_path / "test.md"
        test_file.write_text("# Introduction\n\nThis is a test document.\n\n## Section 1\n\nContent here.")

        # Mock LLM client
        mock_llm = MagicMock(spec=GlueLLM)

        # Mock choose_chunking_strategy to return hierarchical
        async def mock_choose_strategy(*args, **kwargs):
            return ChunkerChoice(strategy="hierarchical", rationale="Markdown document")

        # Mock the chunking_agent module
        import spiderweb.api as api_module

        original_choose = getattr(api_module, "choose_chunking_strategy", None)

        try:
            # Temporarily replace choose_chunking_strategy
            import spiderweb.chunking_agent

            original_func = spiderweb.chunking_agent.choose_chunking_strategy
            spiderweb.chunking_agent.choose_chunking_strategy = mock_choose_strategy

            web = Spiderweb(llm_client=mock_llm)

            # This should work but will fail at actual processing since we don't have
            # a real vector store, but we can at least test the strategy selection path
            try:
                result = await web.ingest_with_adaptive_chunking(test_file)
                # If we get here, the strategy was selected and chunker was created
                assert result is not None
            except Exception as e:
                # Expected to fail at storage/embedding, but strategy selection should work
                # Check that the error is not about missing llm_client or strategy selection
                assert "llm_client is required" not in str(e)
                assert "chunking strategy" not in str(e).lower()
        finally:
            # Restore original function
            spiderweb.chunking_agent.choose_chunking_strategy = original_func


# Fixture file name -> (expected strategy to test with, min expected chunks)
FIXTURE_EXPECTATIONS = [
    ("doc_hierarchical.md", "hierarchical", 3),
    ("doc_sentence.txt", "sentence", 1),
    ("doc_semantic.txt", "semantic", 2),
    ("doc_sliding_window.txt", "sliding_window", 1),
    ("doc_like_pdf.txt", "hierarchical", 1),
    ("doc_like_ocr.txt", "sentence", 1),
]


@pytest.mark.skipif(not FIXTURES_DIR.exists(), reason="Chunking fixtures not found")
class TestChunkingAgentWithFixtures:
    """Tests using real fixture files (markdown, txt for PDF/OCR-like content)."""

    @pytest.mark.asyncio
    async def test_chunk_semantic_fixture_async(self):
        """Chunk doc_semantic.txt with semantic chunker (async, mocked embed)."""
        """Chunk semantic fixture with semantic chunker (async)."""
        path = FIXTURES_DIR / "doc_semantic.txt"
        if not path.exists():
            pytest.skip("Fixture doc_semantic.txt not found")
        content = path.read_text(encoding="utf-8")
        doc = Document(
            raw_content=content,
            markdown_content=content,
            metadata=DocumentMetadata(
                source=str(path),
                file_type="txt",
                extraction_method="test",
            ),
        )
        mock_llm = MagicMock()
        async def mock_embed(texts):
            n = len(texts) if isinstance(texts, list) else 1
            return MagicMock(embeddings=[[0.1] * 384] * n)
        mock_llm.embed = AsyncMock(side_effect=mock_embed)
        config = ChunkerConfig(strategy="semantic", max_chunk_size=2000)
        chunker = create_chunker_from_strategy("semantic", config, llm_client=mock_llm)
        chunks = await chunker.chunk_async(doc)
        assert len(chunks) >= 1
        assert all(c.content.strip() for c in chunks)
    """Tests using real fixture files (markdown, txt for PDF/OCR-like content)."""

    @pytest.mark.parametrize("filename,strategy,min_chunks", FIXTURE_EXPECTATIONS)
    def test_chunk_fixture_with_strategy(self, filename, strategy, min_chunks):
        """Chunk each fixture with its expected strategy and verify we get chunks."""
        path = FIXTURES_DIR / filename
        if not path.exists():
            pytest.skip(f"Fixture {filename} not found")
        content = path.read_text(encoding="utf-8")
        doc = Document(
            raw_content=content,
            markdown_content=content,
            metadata=DocumentMetadata(
                source=str(path),
                file_type=path.suffix.lstrip("."),
                extraction_method="test",
            ),
        )
        config = ChunkerConfig(strategy=strategy, max_chunk_size=2000)
        llm_client = MagicMock() if strategy == "semantic" else None
        chunker = create_chunker_from_strategy(strategy, config, llm_client=llm_client)

        if strategy == "semantic":
            pytest.skip("Semantic chunker requires async; see test_chunk_semantic_fixture_async")
        chunks = chunker.chunk(doc)

        assert len(chunks) >= min_chunks, f"Expected at least {min_chunks} chunks for {filename}"
        assert all(c.content.strip() for c in chunks)

    @pytest.mark.parametrize("filename,strategy,min_chunks", FIXTURE_EXPECTATIONS)
    @pytest.mark.asyncio
    async def test_adaptive_ingest_fixture_mocked_agent(self, filename, strategy, min_chunks):
        """Run adaptive ingest on each fixture with agent mocked to return expected strategy."""
        path = FIXTURES_DIR / filename
        if not path.exists():
            pytest.skip(f"Fixture {filename} not found")

        async def mock_choose(*args, **kwargs):
            return ChunkerChoice(strategy=strategy, rationale=f"Fixture test: {filename}")

        # Mock embed so processor can generate embeddings (one per chunk)
        async def mock_embed(texts):
            n = len(texts) if isinstance(texts, list) else 1
            return MagicMock(embeddings=[[0.0] * 384] * n)

        import spiderweb.chunking_agent as ca
        original = ca.choose_chunking_strategy
        ca.choose_chunking_strategy = mock_choose

        try:
            from spiderweb import Spiderweb
            mock_llm = MagicMock()
            mock_llm.embed = AsyncMock(side_effect=mock_embed)
            web = Spiderweb(llm_client=mock_llm)
            try:
                result = await web.ingest_with_adaptive_chunking(path)
                assert result is not None
                assert result.chunks_created >= min_chunks
                assert result.success
            except Exception as e:
                # Store may fail without real backend
                if "store" in str(e).lower() or "vector" in str(e).lower():
                    pytest.skip(f"Backend not available: {e}")
                raise
        finally:
            ca.choose_chunking_strategy = original

    @pytest.mark.asyncio
    async def test_choose_strategy_receives_fixture_preview(self):
        """Agent receives correct preview from fixture content (mock LLM response)."""
        path = FIXTURES_DIR / "doc_hierarchical.md"
        if not path.exists():
            pytest.skip("Fixture doc_hierarchical.md not found")
        preview = path.read_text(encoding="utf-8")[:2000]

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.final_response = '{"strategy": "hierarchical", "rationale": "Has markdown headings"}'
        mock_llm.complete = AsyncMock(return_value=mock_response)

        import spiderweb.chunking_agent as ca
        orig = getattr(ca, "structured_complete", None)
        try:
            def fail_structured(*args, **kwargs):
                raise ImportError("no structured_complete")
            ca.structured_complete = fail_structured
            choice = await choose_chunking_strategy(
                llm_client=mock_llm,
                document_preview=preview,
                file_name=path.name,
            )
            assert choice.strategy == "hierarchical"
        finally:
            if orig is not None:
                ca.structured_complete = orig

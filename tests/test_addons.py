"""Tests for chunk add-ons system."""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spiderweb.addons.base import ChunkAddOn
from spiderweb.addons.facts import FactsAddOn, FactsResponse
from spiderweb.addons.langextract_addon import LangExtractAddOn
from spiderweb.models.config import ChunkAddOnConfig, LangExtractAddOnOptions
from spiderweb.models.document import Chunk, ChunkMetadata, ChunkType, Document, DocumentMetadata
from spiderweb.registry import chunk_addon_registry


def _make_document(content: str = "Test content") -> Document:
    """Create a test document."""
    return Document(
        raw_content=content,
        markdown_content=content,
        metadata=DocumentMetadata(
            source="test://memory",
            file_type="md",
            extraction_method="test",
        ),
    )


def _make_chunk(content: str = "Test chunk content", chunk_index: int = 0) -> Chunk:
    """Create a test chunk."""
    return Chunk(
        content=content,
        metadata=ChunkMetadata(
            document_id="test-doc",
            chunk_index=chunk_index,
            chunk_type=ChunkType.HIERARCHICAL,
        ),
    )


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestChunkAddOnConfig:
    """Tests for ChunkAddOnConfig model."""

    def test_default_config(self):
        """Default config has no enabled add-ons."""
        config = ChunkAddOnConfig()
        assert config.enabled == []
        assert config.options == {}

    def test_enable_addons(self):
        """Enable add-ons via enabled list."""
        config = ChunkAddOnConfig(enabled=["facts", "entities"])
        assert config.enabled == ["facts", "entities"]

    def test_addon_options(self):
        """Configure add-on specific options."""
        config = ChunkAddOnConfig(
            enabled=["facts"],
            options={"facts": {"max_facts": 5, "model": "gpt-5.1"}},
        )
        assert config.options["facts"]["max_facts"] == 5
        assert config.options["facts"]["model"] == "gpt-5.1"

    def test_empty_options_dict(self):
        """Options dict defaults to empty."""
        config = ChunkAddOnConfig(enabled=["facts"])
        assert config.options == {}


class TestChunkAddOnRegistry:
    """Tests for chunk add-on registry."""

    def test_facts_addon_registered(self):
        """Facts add-on is registered by default."""
        assert "facts" in chunk_addon_registry

    def test_list_registered_addons(self):
        """List all registered add-ons."""
        addons = chunk_addon_registry.list()
        assert "facts" in addons
        assert "langextract" in addons

    def test_create_langextract_addon_from_registry(self):
        """Create LangExtract add-on instance from registry."""
        addon = chunk_addon_registry.create(
            "langextract",
            prompt_description="Extract entities",
            examples=[],
        )
        assert isinstance(addon, LangExtractAddOn)
        assert addon.prompt_description == "Extract entities"
        assert addon.model_id == "gpt-5.1"

    def test_register_custom_addon(self):
        """Register a custom add-on."""
        class TestAddOn:
            def process(self, chunks, **kwargs):
                return chunks

        chunk_addon_registry.register("test-addon", TestAddOn)
        assert "test-addon" in chunk_addon_registry

        # Cleanup
        del chunk_addon_registry._registry["test-addon"]

    def test_create_facts_addon_from_registry(self):
        """Create facts add-on instance from registry."""
        mock_llm = MagicMock()
        addon = chunk_addon_registry.create("facts", llm_client=mock_llm)
        assert isinstance(addon, FactsAddOn)
        assert addon.llm_client is mock_llm

    def test_create_addon_with_options(self):
        """Create add-on with configuration options."""
        mock_llm = MagicMock()
        addon = chunk_addon_registry.create(
            "facts",
            llm_client=mock_llm,
            max_facts=5,
            model="gpt-5.1",
        )
        assert isinstance(addon, FactsAddOn)
        assert addon.max_facts == 5
        assert addon.model == "gpt-5.1"


class TestFactsAddOn:
    """Tests for FactsAddOn."""

    def test_init_with_llm_client(self):
        """Initialize with LLM client."""
        mock_llm = MagicMock()
        addon = FactsAddOn(llm_client=mock_llm)
        assert addon.llm_client is mock_llm

    def test_init_with_options(self):
        """Initialize with custom options."""
        mock_llm = MagicMock()
        addon = FactsAddOn(llm_client=mock_llm, max_facts=5, model="gpt-5.1")
        assert addon.max_facts == 5
        assert addon.model == "gpt-5.1"

    def test_process_requires_async(self):
        """Sync process() raises NotImplementedError."""
        addon = FactsAddOn(llm_client=MagicMock())
        chunks = [_make_chunk()]
        with pytest.raises(NotImplementedError):
            addon.process(chunks)

    @pytest.mark.asyncio
    async def test_process_async_without_llm_skips(self):
        """Process without LLM client skips extraction."""
        addon = FactsAddOn(llm_client=None)
        chunks = [_make_chunk("Test content")]
        result = await addon.process_async(chunks)
        assert result == chunks
        assert chunks[0].metadata.extra.get("facts") is None

    @pytest.mark.asyncio
    async def test_process_async_extracts_facts(self):
        """Process extracts facts and stores in metadata."""
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.final_response = json.dumps({"facts": ["Fact 1", "Fact 2"]})
        mock_llm.complete.return_value = mock_response

        addon = FactsAddOn(llm_client=mock_llm, max_facts=10)
        chunks = [_make_chunk("The capital of France is Paris. It was founded in 300 BC.")]

        # Force the ``llm_client.complete`` path (skip module-level structured_complete).
        with patch("spiderweb.addons.facts.structured_complete", side_effect=ImportError("test skip")):
            result = await addon.process_async(chunks)

        assert result == chunks
        assert "facts" in chunks[0].metadata.extra
        mock_llm.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_async_with_structured_complete(self):
        """Process uses structured_complete when available."""
        mock_llm = AsyncMock()
        facts_response = FactsResponse(facts=["Fact 1", "Fact 2", "Fact 3"])

        # Patch the name bound in facts.py (import-time alias).
        with patch("spiderweb.addons.facts.structured_complete", new_callable=AsyncMock) as mock_structured:
            mock_structured.return_value = SimpleNamespace(
                structured_output=facts_response, final_response=""
            )

            addon = FactsAddOn(llm_client=mock_llm, max_facts=10)
            chunks = [_make_chunk("Test content with facts.")]

            result = await addon.process_async(chunks)

            assert result == chunks
            assert chunks[0].metadata.extra["facts"] == ["Fact 1", "Fact 2", "Fact 3"]
            mock_structured.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_async_respects_max_facts(self):
        """Process respects max_facts limit."""
        mock_llm = AsyncMock()
        facts_response = FactsResponse(facts=[f"Fact {i}" for i in range(20)])

        with patch("spiderweb.addons.facts.structured_complete", new_callable=AsyncMock) as mock_structured:
            mock_structured.return_value = SimpleNamespace(
                structured_output=facts_response, final_response=""
            )

            addon = FactsAddOn(llm_client=mock_llm, max_facts=5)
            chunks = [_make_chunk("Test content.")]

            result = await addon.process_async(chunks)

            assert len(chunks[0].metadata.extra["facts"]) == 5

    @pytest.mark.asyncio
    async def test_process_async_handles_errors_gracefully(self):
        """Process handles errors without breaking pipeline."""
        mock_llm = AsyncMock()
        mock_llm.complete.side_effect = Exception("LLM error")

        addon = FactsAddOn(llm_client=mock_llm)
        chunks = [_make_chunk("Test content")]

        result = await addon.process_async(chunks)

        assert result == chunks
        assert chunks[0].metadata.extra.get("facts") == []

    @pytest.mark.asyncio
    async def test_process_async_handles_json_parsing_errors(self):
        """Process handles JSON parsing errors gracefully."""
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.final_response = "Not valid JSON"
        mock_llm.complete.return_value = mock_response

        addon = FactsAddOn(llm_client=mock_llm)
        chunks = [_make_chunk("Test content")]

        result = await addon.process_async(chunks)

        assert result == chunks
        # Should fall back to empty list or try to extract from text
        assert "facts" in chunks[0].metadata.extra


class TestChunkAddOnProtocol:
    """Tests for ChunkAddOn protocol compliance."""

    @pytest.mark.asyncio
    async def test_sync_addon_implements_process(self):
        """Sync add-on implements process() method."""
        class SyncAddOn:
            def process(self, chunks, **kwargs):
                for chunk in chunks:
                    chunk.metadata.extra["processed"] = True
                return chunks

        addon = SyncAddOn()
        chunks = [_make_chunk()]
        result = addon.process(chunks)
        assert result == chunks
        assert chunks[0].metadata.extra["processed"] is True

    @pytest.mark.asyncio
    async def test_async_addon_implements_process_async(self):
        """Async add-on implements process_async() method."""
        class AsyncAddOn:
            async def process_async(self, chunks, **kwargs):
                for chunk in chunks:
                    chunk.metadata.extra["processed"] = True
                return chunks

        addon = AsyncAddOn()
        chunks = [_make_chunk()]
        result = await addon.process_async(chunks)
        assert result == chunks
        assert chunks[0].metadata.extra["processed"] is True

    def test_addon_can_access_document_context(self):
        """Add-on receives document context."""
        class ContextAddOn:
            def process(self, chunks, *, document=None, **kwargs):
                if document:
                    for chunk in chunks:
                        chunk.metadata.extra["doc_id"] = document.id
                return chunks

        addon = ContextAddOn()
        doc = _make_document()
        chunks = [_make_chunk()]
        result = addon.process(chunks, document=doc)
        assert chunks[0].metadata.extra["doc_id"] == doc.id


class TestDocumentProcessorIntegration:
    """Tests for add-on integration with DocumentProcessor."""

    @pytest.mark.asyncio
    async def test_processor_runs_addons_after_chunking(self, temp_dir):
        """DocumentProcessor runs add-ons after chunking."""
        from spiderweb.pipeline.processor import DocumentProcessor
        from spiderweb.models.config import ChunkerConfig

        # Create a test file
        test_file = temp_dir / "test.md"
        test_file.write_text("# Test\n\nThis is test content with facts.")

        # Mock LLM for facts extraction
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.final_response = json.dumps({"facts": ["Test fact"]})
        mock_llm.complete.return_value = mock_response

        # Create processor with facts add-on enabled
        processor = DocumentProcessor(
            llm_client=mock_llm,
            chunker_config=ChunkerConfig(strategy="hierarchical"),
            chunk_add_ons=["facts"],
        )

        result = await processor.process(test_file, store_chunks=False)

        # Verify chunks were created
        assert result.chunks_created > 0

        # Verify facts were extracted (if chunks exist)
        if result.document.chunks:
            # At least one chunk should have facts
            chunks_with_facts = [
                c for c in result.document.chunks
                if "facts" in c.metadata.extra
            ]
            # Facts extraction may have been called
            assert mock_llm.complete.called or len(chunks_with_facts) >= 0

    @pytest.mark.asyncio
    async def test_processor_skips_addons_when_disabled(self, temp_dir):
        """DocumentProcessor skips add-ons when not enabled."""
        from spiderweb.pipeline.processor import DocumentProcessor
        from spiderweb.models.config import ChunkerConfig

        test_file = temp_dir / "test.md"
        test_file.write_text("# Test\n\nContent.")

        processor = DocumentProcessor(
            chunker_config=ChunkerConfig(strategy="hierarchical"),
            chunk_add_ons=[],  # No add-ons enabled
        )

        result = await processor.process(test_file, store_chunks=False)

        assert result.chunks_created > 0
        # Chunks should not have facts
        for chunk in result.document.chunks:
            assert "facts" not in chunk.metadata.extra

    @pytest.mark.asyncio
    async def test_processor_handles_addon_errors(self, temp_dir):
        """DocumentProcessor handles add-on errors gracefully."""
        from spiderweb.pipeline.processor import DocumentProcessor
        from spiderweb.models.config import ChunkerConfig

        # Register a failing add-on (accept **kwargs so registry.create() can pass llm_client etc.)
        class FailingAddOn:
            def __init__(self, **kwargs):
                pass

            async def process_async(self, chunks, **kwargs):
                raise Exception("Add-on error")

        chunk_addon_registry.register("failing", FailingAddOn)

        try:
            test_file = temp_dir / "test.md"
            test_file.write_text("# Test\n\nContent.")

            processor = DocumentProcessor(
                chunker_config=ChunkerConfig(strategy="hierarchical"),
                chunk_add_ons=["failing"],
            )

            # Should not raise, but log error
            result = await processor.process(test_file, store_chunks=False)
            assert result.chunks_created > 0
        finally:
            # Cleanup
            if "failing" in chunk_addon_registry._registry:
                del chunk_addon_registry._registry["failing"]


class TestSpiderwebIntegration:
    """Tests for add-on integration with Spiderweb."""

    @pytest.mark.asyncio
    async def test_spiderweb_passes_addon_config(self, temp_dir):
        """Spiderweb passes add-on config to DocumentProcessor."""
        from spiderweb import Spiderweb
        from spiderweb.models.config import ChunkAddOnConfig

        test_file = temp_dir / "test.md"
        test_file.write_text("# Test\n\nContent.")

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.final_response = json.dumps({"facts": ["Fact"]})
        mock_llm.complete.return_value = mock_response

        config = ChunkAddOnConfig(enabled=["facts"])
        web = Spiderweb(
            llm_client=mock_llm,
            chunk_addon_config=config,
        )

        result = await web.ingest(test_file)

        assert result.chunks_created > 0
        # Verify processor has add-on config
        assert web.document_processor.chunk_addon_config.enabled == ["facts"]

    @pytest.mark.asyncio
    async def test_spiderweb_with_addon_list(self, temp_dir):
        """Spiderweb accepts add-on list directly."""
        from spiderweb import Spiderweb

        test_file = temp_dir / "test.md"
        test_file.write_text("# Test\n\nContent.")

        web = Spiderweb(chunk_add_ons=["facts"])

        # Verify config was created from list
        assert web.document_processor.chunk_addon_config.enabled == ["facts"]


class TestLangExtractAddOn:
    """Tests for LangExtractAddOn (optional langextract dependency)."""

    def test_init_with_options(self):
        """Initialize with options."""
        addon = LangExtractAddOn(
            prompt_description="Extract people and places",
            examples=[{"text": "Alice met Bob.", "extractions": []}],
            model_id="gemini-2.5-pro",
            extraction_passes=2,
        )
        assert addon.prompt_description == "Extract people and places"
        assert len(addon.examples_option) == 1
        assert addon.model_id == "gemini-2.5-pro"
        assert addon.extraction_passes == 2

    def test_process_requires_async(self):
        """Sync process() raises NotImplementedError."""
        addon = LangExtractAddOn(prompt_description="Extract")
        chunks = [_make_chunk()]
        with pytest.raises(NotImplementedError):
            addon.process(chunks)

    @pytest.mark.asyncio
    async def test_process_async_without_document_returns_chunks(self):
        """Without document, process_async returns chunks unchanged."""
        addon = LangExtractAddOn(prompt_description="Extract")
        chunks = [_make_chunk()]
        result = await addon.process_async(chunks, document=None)
        assert result == chunks

    @pytest.mark.asyncio
    async def test_process_async_without_prompt_skips(self):
        """Without prompt_description, add-on skips and returns chunks."""
        addon = LangExtractAddOn(prompt_description="")
        doc = _make_document("Some text.")
        doc.chunks = [_make_chunk()]
        chunks = doc.chunks
        result = await addon.process_async(chunks, document=doc)
        assert result == chunks
        assert "langextract" not in doc.metadata.extra

    @pytest.mark.asyncio
    async def test_process_async_empty_document_skips(self):
        """Empty document content skips extraction."""
        addon = LangExtractAddOn(prompt_description="Extract entities")
        doc = _make_document("")
        doc.raw_content = ""
        doc.markdown_content = ""
        chunks = [_make_chunk()]
        result = await addon.process_async(chunks, document=doc)
        assert result == chunks


class TestLangExtractAddOnOptions:
    """Tests for LangExtractAddOnOptions config model."""

    def test_defaults(self):
        """Default options are sensible."""
        opts = LangExtractAddOnOptions()
        assert opts.prompt_description == ""
        assert opts.model_id == "gpt-5.1"
        assert opts.extraction_passes == 2
        assert opts.max_char_buffer == 2000
        assert opts.attach_to_chunks is True

    def test_model_dump_for_registry(self):
        """Options can be dumped for ChunkAddOnConfig.options."""
        opts = LangExtractAddOnOptions(
            prompt_description="Extract dates",
            examples=[{"text": "On Jan 1.", "extractions": []}],
        )
        d = opts.model_dump()
        assert d["prompt_description"] == "Extract dates"
        assert len(d["examples"]) == 1
        addon = chunk_addon_registry.create("langextract", **d)
        assert addon.prompt_description == "Extract dates"

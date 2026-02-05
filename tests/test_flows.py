"""End-to-end flow tests for Spiderweb.

These tests verify that the major processing flows work correctly
when components are wired together. Uses mocks for external services
(LLMs, crawl4ai) but tests real integration between internal components.
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spiderweb.chunkers.hierarchical import HierarchicalChunker
from spiderweb.chunkers.sentence import SentenceChunker
from spiderweb.crawlers.base import CrawlResult
from spiderweb.models.config import (
    ChunkerConfig,
    CrawlerConfig,
    SearchDepthConfig,
    SearchProviderConfig,
    ValidatorConfig,
    VectorStoreConfig,
)
from spiderweb.models.document import ChunkType, Document, DocumentMetadata
from spiderweb.stores.memory import MemoryVectorStore


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_markdown():
    """Sample markdown content for testing."""
    return """# Introduction

This is the introduction paragraph with important information.

## Section One

Here we discuss the first topic in detail. This section contains
multiple sentences that should be chunked appropriately.

## Section Two

The second section covers another topic. It has its own content
that is distinct from section one.

### Subsection 2.1

A nested subsection with more specific information.

## Conclusion

Final thoughts and summary of the document.
"""


@pytest.fixture
def sample_document(sample_markdown):
    """Create a sample Document for testing."""
    return Document(
        raw_content=sample_markdown,
        markdown_content=sample_markdown,
        metadata=DocumentMetadata(
            source="test://sample.md",
            file_type="md",
            extraction_method="test",
        ),
    )


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client for testing."""
    mock = MagicMock()
    
    # Mock embedding response
    mock_embed_response = MagicMock()
    mock_embed_response.embeddings = [[0.1] * 1536]
    mock.embed = AsyncMock(return_value=mock_embed_response)
    
    # Mock generate response
    mock_generate_response = MagicMock()
    mock_generate_response.content = "Generated response"
    mock_generate_response.text = "Generated response"
    mock.generate = AsyncMock(return_value=mock_generate_response)
    
    return mock


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# Document Processing Flow Tests
# =============================================================================


class TestDocumentChunkingFlow:
    """Test document → chunking flow."""

    def test_hierarchical_chunking_preserves_structure(self, sample_document):
        """Hierarchical chunker preserves document structure."""
        chunker = HierarchicalChunker(max_chunk_size=500)
        
        chunks = chunker.chunk(sample_document)
        
        # Should create multiple chunks
        assert len(chunks) >= 3
        
        # Each chunk should have metadata
        for chunk in chunks:
            assert chunk.metadata.document_id == sample_document.id
            assert chunk.metadata.chunk_type == ChunkType.HIERARCHICAL
        
        # First chunk should be the intro/title
        assert "Introduction" in chunks[0].content or "# " in chunks[0].content
        
        # Parent-child relationships should be set
        section_chunks = [c for c in chunks if c.metadata.section_title]
        assert len(section_chunks) > 0

    def test_sentence_chunking_creates_logical_breaks(self, sample_document):
        """Sentence chunker creates chunks at logical boundaries."""
        chunker = SentenceChunker(max_chunk_size=200, min_chunk_size=50)
        
        chunks = chunker.chunk(sample_document)
        
        # Should create multiple chunks
        assert len(chunks) >= 2
        
        # Each chunk should have sequential indices
        indices = [c.metadata.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))
        
        # Chunks should not exceed max size (approximately)
        for chunk in chunks:
            # Allow some overflow for sentence boundaries
            assert len(chunk.content) <= 400

    def test_chunking_with_validation(self, sample_document, mock_llm_client):
        """Chunking followed by validation filters low-quality chunks."""
        from spiderweb.validators.pipeline import ValidationPipeline
        
        chunker = HierarchicalChunker(max_chunk_size=500)
        chunks = chunker.chunk(sample_document)
        
        # Add a low-quality chunk
        from spiderweb.models.document import Chunk, ChunkMetadata
        bad_chunk = Chunk(
            content="----....----",  # Low information density
            metadata=ChunkMetadata(
                document_id=sample_document.id,
                chunk_index=len(chunks),
                chunk_type=ChunkType.CUSTOM,
            ),
        )
        chunks.append(bad_chunk)
        
        # Validate
        pipeline = ValidationPipeline(llm_client=None)
        
        async def validate():
            return await pipeline.validate_batch(chunks)
        
        import asyncio
        results = asyncio.run(validate())
        
        # Most chunks should pass, but bad chunk should fail
        passed = [r for r in results if r.passed]
        failed = [r for r in results if not r.passed]
        
        assert len(passed) >= 3
        assert len(failed) >= 1


class TestVectorStoreFlow:
    """Test chunk → embed → store → query flow."""

    async def test_memory_store_upsert_and_query(self, sample_document, mock_llm_client):
        """Chunks can be stored and queried from memory store."""
        from spiderweb.models.document import Chunk, ChunkMetadata
        
        store = MemoryVectorStore()
        
        # Create chunks with embeddings (non-zero for valid cosine similarity)
        chunks = []
        embeddings = [
            [1.0, 0.1, 0.1] + [0.1] * 1533,  # Python programming
            [0.1, 1.0, 0.1] + [0.1] * 1533,  # Web scraping
            [0.1, 0.1, 1.0] + [0.1] * 1533,  # Data analysis
        ]
        for i, content in enumerate(["Python programming", "Web scraping", "Data analysis"]):
            chunk = Chunk(
                content=content,
                metadata=ChunkMetadata(
                    document_id=sample_document.id,
                    chunk_index=i,
                    chunk_type=ChunkType.SENTENCE,
                ),
            )
            chunk.embedding = embeddings[i]
            chunks.append(chunk)
        
        # Upsert
        await store.upsert(chunks)
        
        # Query with embedding similar to "Python programming"
        query_embedding = [1.0, 0.1, 0.1] + [0.1] * 1533
        results = await store.query(query_embedding, top_k=2)
        
        # Should return results ordered by similarity
        assert len(results) == 2
        assert results[0][0].content == "Python programming"
        assert results[0][1] > results[1][1]  # Higher score for better match

    async def test_document_to_store_full_flow(self, sample_document, mock_llm_client):
        """Full flow: document → chunk → embed → store."""
        from spiderweb.models.document import Chunk, ChunkMetadata
        
        # Chunk
        chunker = HierarchicalChunker(max_chunk_size=200)
        chunks = chunker.chunk(sample_document)
        
        assert len(chunks) > 0
        
        # Add embeddings (mock)
        for i, chunk in enumerate(chunks):
            # Create distinct embeddings based on content hash
            chunk.embedding = [float(hash(chunk.content) % 100) / 100.0] + [0.1] * 1535
        
        # Store
        store = MemoryVectorStore()
        await store.upsert(chunks)
        
        # Query
        query_embedding = chunks[0].embedding  # Query with first chunk's embedding
        results = await store.query(query_embedding, top_k=3)
        
        # Should find the original chunk
        assert len(results) >= 1
        assert results[0][0].id == chunks[0].id
        assert results[0][1] == pytest.approx(1.0, abs=0.01)  # Perfect match


# =============================================================================
# Crawl Flow Tests
# =============================================================================


class TestCrawlProcessingFlow:
    """Test crawl → process → chunk flow."""

    def test_crawl_result_to_document_conversion(self):
        """CrawlResult converts to Document correctly."""
        from spiderweb.loaders.web_loader import WebLoader
        
        crawl_result = CrawlResult(
            url="https://example.com/article",
            content="<html><body><h1>Title</h1><p>Content</p></body></html>",
            markdown="# Title\n\nContent paragraph here.",
            status_code=200,
            success=True,
            links=["https://example.com/link1"],
            metadata={"title": "Test Article"},
        )
        
        loader = WebLoader()
        document = loader._crawl_result_to_document(crawl_result)
        
        assert document.markdown_content == "# Title\n\nContent paragraph here."
        assert document.metadata.source == "https://example.com/article"
        assert document.metadata.file_type == "html"
        assert document.metadata.extra["status_code"] == 200

    def test_crawl_to_chunk_flow(self):
        """Full flow: crawl result → document → chunks."""
        from spiderweb.loaders.web_loader import WebLoader
        
        crawl_result = CrawlResult(
            url="https://example.com/article",
            content="<html></html>",
            markdown="""# Article Title

First paragraph with important content.

## Section One

Details about section one topic.

## Section Two

More information in section two.
""",
            status_code=200,
            success=True,
        )
        
        loader = WebLoader()
        document = loader._crawl_result_to_document(crawl_result)
        
        chunker = HierarchicalChunker(max_chunk_size=500)
        chunks = chunker.chunk(document)
        
        assert len(chunks) >= 2
        
        # Check document ID propagated
        for chunk in chunks:
            assert chunk.metadata.document_id == document.id


class TestCrawlStorageFlow:
    """Test crawl → save to disk flow."""

    def test_crawl_save_and_index_flow(self, temp_dir):
        """Full flow: crawl results → save → create index."""
        from spiderweb.crawlers.storage import CrawlStorage
        
        storage = CrawlStorage(output_dir=temp_dir)
        
        # Save multiple crawl results
        results = [
            CrawlResult(
                url="https://example.com/page1",
                content="<html>Page 1</html>",
                markdown="# Page 1",
                status_code=200,
                success=True,
            ),
            CrawlResult(
                url="https://example.com/page2",
                content="<html>Page 2</html>",
                markdown="# Page 2",
                status_code=200,
                success=True,
            ),
        ]
        
        for result in results:
            storage.save_crawl_result(result, format="all")
        
        # Create index
        index_path = storage.create_index()
        
        assert index_path.exists()
        
        # Verify files were created
        import json
        index_data = json.loads(index_path.read_text())
        
        # Should have 6 files (3 per result: .md, .html, .json)
        assert len(index_data["files"]) == 6


# =============================================================================
# Search Flow Tests
# =============================================================================


class TestSearchProviderFlow:
    """Test search provider integration."""

    async def test_stub_search_provider_returns_results(self):
        """Stub search provider returns mock results."""
        from spiderweb.search.stub import StubSearchProvider
        
        provider = StubSearchProvider()
        results = await provider.search("test query", limit=5)
        
        assert len(results.results) > 0
        assert results.query == "test query"
        
        # Each result should have required fields
        for result in results.results:
            assert result.url
            assert result.title


class TestSearchCrawlFlow:
    """Test search → crawl → extract flow."""

    async def test_search_to_crawl_flow_with_mocks(self, mock_llm_client, temp_dir):
        """Simulated search → crawl flow with mocked external services."""
        from spiderweb.search.stub import StubSearchProvider
        from spiderweb.search.base import SearchResult
        
        # Search
        provider = StubSearchProvider()
        search_results = await provider.search("python tutorial", limit=3)
        
        assert len(search_results.results) > 0
        
        # Simulate crawling each result
        crawl_results = []
        for sr in search_results.results:
            # In real usage, this would call crawler.crawl(sr.url)
            crawl_result = CrawlResult(
                url=sr.url,
                content=f"<html><body>{sr.title}</body></html>",
                markdown=f"# {sr.title}\n\nContent from {sr.url}",
                status_code=200,
                success=True,
            )
            crawl_results.append(crawl_result)
        
        # Save to storage
        from spiderweb.crawlers.storage import CrawlStorage
        storage = CrawlStorage(output_dir=temp_dir)
        
        for result in crawl_results:
            storage.save_crawl_result(result, format="json")
        
        # Verify files were created
        json_files = list(temp_dir.glob("*.json"))
        assert len(json_files) == len(search_results.results)


# =============================================================================
# Query Expansion Flow Tests
# =============================================================================


class TestQueryExpansionFlow:
    """Test query expansion → search → fusion flow."""

    async def test_multi_query_to_rrf_flow(self, mock_llm_client):
        """Query expansion with RRF fusion."""
        from spiderweb.models.config import QueryExpansionConfig
        from spiderweb.pipeline.query_expansion import QueryExpander, reciprocal_rank_fusion
        from spiderweb.models.document import Chunk, ChunkMetadata
        
        # Setup mock to return expanded queries
        mock_llm_client.generate.return_value.content = """What is machine learning?
How does ML work?
Explain machine learning basics"""
        
        config = QueryExpansionConfig(
            enabled=True,
            strategy="multi_query",
            num_expansions=3,
            include_original=True,
        )
        
        expander = QueryExpander(mock_llm_client, config)
        queries = await expander.expand("What is ML?")
        
        assert len(queries) == 4  # Original + 3 expanded
        
        # Simulate search results for each query
        chunk1 = Chunk(
            content="Machine learning is...",
            metadata=ChunkMetadata(document_id="doc1", chunk_index=0, chunk_type=ChunkType.SENTENCE),
        )
        chunk2 = Chunk(
            content="ML algorithms can...",
            metadata=ChunkMetadata(document_id="doc2", chunk_index=0, chunk_type=ChunkType.SENTENCE),
        )
        
        # Different rankings per query
        query_results = [
            [(chunk1, 0.9), (chunk2, 0.7)],
            [(chunk2, 0.95), (chunk1, 0.6)],
            [(chunk1, 0.85), (chunk2, 0.8)],
            [(chunk1, 0.88), (chunk2, 0.75)],
        ]
        
        # Apply RRF
        fused = reciprocal_rank_fusion(query_results, k=60)
        
        # chunk1 appears at rank 1 in 3/4 queries, should rank higher
        assert len(fused) == 2
        assert fused[0][0].id == chunk1.id


# =============================================================================
# Full Pipeline Tests
# =============================================================================


class TestFullPipeline:
    """Test complete end-to-end pipelines."""

    def test_document_ingestion_pipeline(self, temp_dir):
        """Full ingestion: file → extract → chunk → validate → ready for store."""
        from spiderweb.models.document import Document, DocumentMetadata
        from spiderweb.chunkers.hierarchical import HierarchicalChunker
        from spiderweb.validators.quality import QualityValidator
        
        # Create a test file
        test_file = temp_dir / "test_doc.md"
        test_file.write_text("""# Test Document

This is a test document with meaningful content.

## Section One

This section contains important information about topic one.
It has multiple sentences to ensure proper chunking.

## Section Two

Another section with different content.
This helps test the hierarchical chunking.
""")
        
        # Simulate extraction (in real usage, would use MarkitdownExtractor)
        document = Document(
            raw_content=test_file.read_text(),
            markdown_content=test_file.read_text(),
            metadata=DocumentMetadata(
                source=str(test_file),
                file_type="md",
                extraction_method="test",
            ),
        )
        
        # Chunk
        chunker = HierarchicalChunker(max_chunk_size=300)
        chunks = chunker.chunk(document)
        document.chunks = chunks
        
        assert len(chunks) >= 2
        
        # Validate
        validator = QualityValidator(min_score=0.3)
        
        async def validate_all():
            results = []
            for chunk in chunks:
                result = await validator.validate(chunk)
                results.append(result)
            return results
        
        import asyncio
        validation_results = asyncio.run(validate_all())
        
        # All should pass (meaningful content)
        passed = [r for r in validation_results if r.passed]
        assert len(passed) == len(chunks)

    async def test_query_pipeline(self, mock_llm_client):
        """Full query: embed query → search store → return results."""
        from spiderweb.models.document import Chunk, ChunkMetadata
        
        # Setup: Create and populate store
        store = MemoryVectorStore()
        
        chunks = [
            Chunk(
                content="Python is a programming language",
                metadata=ChunkMetadata(document_id="doc1", chunk_index=0, chunk_type=ChunkType.SENTENCE),
            ),
            Chunk(
                content="JavaScript runs in browsers",
                metadata=ChunkMetadata(document_id="doc2", chunk_index=0, chunk_type=ChunkType.SENTENCE),
            ),
            Chunk(
                content="Python is great for data science",
                metadata=ChunkMetadata(document_id="doc1", chunk_index=1, chunk_type=ChunkType.SENTENCE),
            ),
        ]
        
        # Add embeddings (mock semantic similarity)
        chunks[0].embedding = [1.0, 0.0, 0.0] + [0.0] * 1533  # Python
        chunks[1].embedding = [0.0, 1.0, 0.0] + [0.0] * 1533  # JavaScript
        chunks[2].embedding = [0.9, 0.1, 0.5] + [0.0] * 1533  # Python + data
        
        await store.upsert(chunks)
        
        # Query: "What is Python?"
        query_embedding = [1.0, 0.0, 0.0] + [0.0] * 1533  # Similar to Python chunks
        
        results = await store.query(query_embedding, top_k=2)
        
        # Should return Python-related chunks
        assert len(results) == 2
        assert "Python" in results[0][0].content
        assert "Python" in results[1][0].content
        assert results[0][1] > results[1][1]  # First should have higher score

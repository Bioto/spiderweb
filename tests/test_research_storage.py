"""Tests for research content store (MarkdownResearchStore and protocol)."""

import tempfile
from pathlib import Path

import pytest

from spiderweb.research.storage import MarkdownResearchStore, ResearchContentStore


@pytest.fixture
def temp_dir():
    """Temporary directory for store tests."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


class TestMarkdownResearchStoreInit:
    """Tests for MarkdownResearchStore initialization."""

    def test_creates_base_directory(self, temp_dir):
        """Store creates base directory if it does not exist."""
        base = temp_dir / "cache" / "run_1"
        assert not base.exists()
        store = MarkdownResearchStore(base)
        assert base.exists()
        assert base.is_dir()

    def test_creates_subdir_when_given(self, temp_dir):
        """Store creates subdirectory under base when subdir is provided."""
        store = MarkdownResearchStore(temp_dir, subdir="query_slug")
        sub = temp_dir / "query_slug"
        assert sub.exists()
        assert sub.is_dir()

    def test_accepts_string_path(self, temp_dir):
        """Store accepts string path for base_dir."""
        store = MarkdownResearchStore(str(temp_dir))
        assert store.base_dir == temp_dir


class TestMarkdownResearchStoreSaveAndGet:
    """Tests for save_page_content and get_content."""

    def test_save_returns_absolute_path_ref(self, temp_dir):
        """save_page_content returns absolute path as ref."""
        store = MarkdownResearchStore(temp_dir)
        ref = store.save_page_content("https://example.com/page", "# Hello\n\nWorld.")
        assert Path(ref).is_absolute()
        assert ref.endswith(".md")

    def test_get_content_returns_saved_content(self, temp_dir):
        """get_content returns the content that was saved."""
        store = MarkdownResearchStore(temp_dir)
        content = "# Example\n\nThis is **markdown**."
        ref = store.save_page_content("https://example.com/doc", content)
        retrieved = store.get_content(ref)
        assert retrieved == content

    def test_save_with_metadata_stores_frontmatter(self, temp_dir):
        """Saving with metadata writes YAML frontmatter; get_content strips it."""
        store = MarkdownResearchStore(temp_dir)
        ref = store.save_page_content(
            "https://example.com/page",
            "Body text here.",
            metadata={"source_query": "test query", "status_code": 200},
        )
        raw = Path(ref).read_text()
        assert raw.startswith("---")
        assert "source_query: test query" in raw
        assert "Body text here." in raw
        retrieved = store.get_content(ref)
        assert retrieved == "Body text here."

    def test_get_content_missing_file_returns_none(self, temp_dir):
        """get_content returns None for missing path."""
        store = MarkdownResearchStore(temp_dir)
        result = store.get_content(str(temp_dir / "nonexistent.md"))
        assert result is None

    def test_multiple_pages_under_subdir(self, temp_dir):
        """Multiple pages can be saved under same subdir; each has unique ref."""
        store = MarkdownResearchStore(temp_dir, subdir="run_1")
        ref1 = store.save_page_content("https://a.com/1", "Content 1")
        ref2 = store.save_page_content("https://b.com/2", "Content 2")
        assert ref1 != ref2
        assert store.get_content(ref1) == "Content 1"
        assert store.get_content(ref2) == "Content 2"

    def test_protocol_compliance(self, temp_dir):
        """MarkdownResearchStore satisfies ResearchContentStore protocol."""
        store = MarkdownResearchStore(temp_dir)
        assert isinstance(store, ResearchContentStore)
        ref = store.save_page_content("https://example.com/p", "x")
        assert isinstance(ref, str)
        assert store.get_content(ref) == "x"

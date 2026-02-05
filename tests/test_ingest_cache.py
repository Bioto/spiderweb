"""Tests for IngestCache functionality.

Tests file hash tracking, modification time checking, persistence,
and cache management operations.
"""

import json
import tempfile
from pathlib import Path

import pytest

from spiderweb.utils.ingest_cache import IngestCache


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def cache_file(temp_dir):
    """Create a unique cache file path for each test."""
    return temp_dir / "test_cache.json"


@pytest.fixture
def test_file(temp_dir):
    """Create a test file with known content."""
    file_path = temp_dir / "test.txt"
    file_path.write_text("Hello, world!")
    return file_path


class TestIngestCacheShouldProcess:
    """Tests for should_process method."""

    def test_should_process_first_time(self, cache_file, test_file):
        """First time checking a file returns True."""
        cache = IngestCache(cache_file=cache_file)
        assert cache.should_process(test_file) is True

    def test_should_process_after_update(self, cache_file, test_file):
        """After updating cache, same file returns False."""
        cache = IngestCache(cache_file=cache_file)
        assert cache.should_process(test_file) is True
        
        cache.update(test_file, "doc-123")
        assert cache.should_process(test_file) is False

    def test_should_process_after_content_change(self, cache_file, test_file):
        """After file content changes, should_process returns True."""
        cache = IngestCache(cache_file=cache_file)
        cache.update(test_file, "doc-123")
        assert cache.should_process(test_file) is False
        
        # Change file content
        test_file.write_text("Modified content")
        assert cache.should_process(test_file) is True

    def test_should_process_force_true(self, cache_file, test_file):
        """force=True always returns True."""
        cache = IngestCache(cache_file=cache_file)
        cache.update(test_file, "doc-123")
        
        # Even though file is cached, force=True bypasses cache
        assert cache.should_process(test_file, force=True) is True

    def test_should_process_nonexistent_file(self, cache_file, temp_dir):
        """Non-existent file returns False."""
        cache = IngestCache(cache_file=cache_file)
        nonexistent = temp_dir / "nonexistent.txt"
        assert cache.should_process(nonexistent) is False


class TestIngestCacheUpdate:
    """Tests for update method."""

    def test_update_stores_hash_mtime_document_id(self, cache_file, test_file):
        """Update stores hash, mtime, and document_id."""
        cache = IngestCache(cache_file=cache_file)
        cache.update(test_file, "doc-123")
        
        file_str = str(test_file.resolve())
        entry = cache._cache[file_str]
        
        assert "hash" in entry
        assert "mtime" in entry
        assert entry["document_id"] == "doc-123"
        assert len(entry["hash"]) == 64  # SHA256 hex digest length

    def test_update_nonexistent_file(self, cache_file, temp_dir):
        """Update of non-existent file doesn't crash and doesn't add entry."""
        cache = IngestCache(cache_file=cache_file)
        nonexistent = temp_dir / "nonexistent.txt"
        
        cache.update(nonexistent, "doc-123")
        
        file_str = str(nonexistent.resolve())
        assert file_str not in cache._cache


class TestIngestCacheGetDocumentId:
    """Tests for get_document_id method."""

    def test_get_document_id_returns_stored_id(self, cache_file, test_file):
        """Returns stored document_id for cached file."""
        cache = IngestCache(cache_file=cache_file)
        cache.update(test_file, "doc-456")
        
        assert cache.get_document_id(test_file) == "doc-456"

    def test_get_document_id_returns_none_for_uncached(self, cache_file, test_file):
        """Returns None for uncached file."""
        cache = IngestCache(cache_file=cache_file)
        assert cache.get_document_id(test_file) is None


class TestIngestCacheRemove:
    """Tests for remove method."""

    def test_remove_deletes_entry(self, cache_file, test_file):
        """Remove deletes cache entry."""
        cache = IngestCache(cache_file=cache_file)
        cache.update(test_file, "doc-123")
        assert cache.get_document_id(test_file) == "doc-123"
        
        cache.remove(test_file)
        assert cache.get_document_id(test_file) is None

    def test_remove_nonexistent_entry(self, cache_file, test_file):
        """Removing non-existent entry doesn't crash."""
        cache = IngestCache(cache_file=cache_file)
        cache.remove(test_file)  # Should not raise


class TestIngestCacheClear:
    """Tests for clear method."""

    def test_clear_removes_all_entries(self, cache_file, temp_dir):
        """Clear removes all cache entries."""
        cache = IngestCache(cache_file=cache_file)
        
        # Add multiple files
        file1 = temp_dir / "file1.txt"
        file1.write_text("Content 1")
        file2 = temp_dir / "file2.txt"
        file2.write_text("Content 2")
        
        cache.update(file1, "doc-1")
        cache.update(file2, "doc-2")
        
        assert len(cache._cache) == 2
        
        cache.clear()
        assert len(cache._cache) == 0

    def test_should_process_after_clear(self, cache_file, test_file):
        """After clear, should_process returns True for previously cached file."""
        cache = IngestCache(cache_file=cache_file)
        cache.update(test_file, "doc-123")
        assert cache.should_process(test_file) is False
        
        cache.clear()
        assert cache.should_process(test_file) is True


class TestIngestCachePersistence:
    """Tests for save/load persistence."""

    def test_save_and_load(self, cache_file, test_file):
        """Cache persists across instances."""
        cache1 = IngestCache(cache_file=cache_file)
        cache1.update(test_file, "doc-123")
        cache1.save()
        
        # Create new instance, should load same data
        cache2 = IngestCache(cache_file=cache_file)
        assert cache2.get_document_id(test_file) == "doc-123"
        assert cache2.should_process(test_file) is False

    def test_load_corrupt_file(self, cache_file):
        """Corrupt cache file loads as empty cache."""
        # Write invalid JSON
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text("invalid json{")
        
        cache = IngestCache(cache_file=cache_file)
        # Should not crash, cache should be empty
        assert len(cache._cache) == 0

    def test_load_missing_file(self, cache_file):
        """Missing cache file loads as empty cache."""
        cache = IngestCache(cache_file=cache_file)
        assert len(cache._cache) == 0

    def test_save_creates_directory(self, temp_dir):
        """Save creates parent directory if it doesn't exist."""
        cache_file = temp_dir / "nested" / "deep" / "cache.json"
        cache = IngestCache(cache_file=cache_file)
        
        test_file = temp_dir / "test.txt"
        test_file.write_text("test")
        cache.update(test_file, "doc-1")
        
        cache.save()
        assert cache_file.exists()


class TestIngestCacheIsolation:
    """Tests for test isolation."""

    def test_multiple_instances_independent(self, temp_dir):
        """Multiple cache instances with different files are independent."""
        cache_file1 = temp_dir / "cache1.json"
        cache_file2 = temp_dir / "cache2.json"
        
        file1 = temp_dir / "file1.txt"
        file1.write_text("Content 1")
        file2 = temp_dir / "file2.txt"
        file2.write_text("Content 2")
        
        cache1 = IngestCache(cache_file=cache_file1)
        cache1.update(file1, "doc-1")
        
        cache2 = IngestCache(cache_file=cache_file2)
        cache2.update(file2, "doc-2")
        
        # Each cache only knows about its own file
        assert cache1.get_document_id(file1) == "doc-1"
        assert cache1.get_document_id(file2) is None
        
        assert cache2.get_document_id(file2) == "doc-2"
        assert cache2.get_document_id(file1) is None

"""Tests for path utility functions.

These tests ensure that search queries are safely converted to
filesystem-compatible paths for storing crawled content.
"""

import pytest

from spiderweb.utils.path_utils import sanitize_query_for_path


class TestSanitizeQueryForPath:
    """Tests for sanitize_query_for_path function."""

    def test_simple_query(self):
        """Simple queries are lowercased and spaces become underscores."""
        assert sanitize_query_for_path("Python Web Scraping") == "python_web_scraping"

    def test_already_safe(self):
        """Already safe strings pass through unchanged except lowercase."""
        assert sanitize_query_for_path("hello_world") == "hello_world"

    def test_special_characters_replaced(self):
        """Special characters are replaced with underscores."""
        assert sanitize_query_for_path("what is 2+2?") == "what_is_2_2"
        assert sanitize_query_for_path("search: foo/bar") == "search_foo_bar"
        assert sanitize_query_for_path("C++ programming") == "c_programming"

    def test_multiple_spaces_collapsed(self):
        """Multiple consecutive spaces become a single underscore."""
        assert sanitize_query_for_path("hello    world") == "hello_world"

    def test_multiple_underscores_collapsed(self):
        """Multiple consecutive underscores are collapsed."""
        assert sanitize_query_for_path("hello___world") == "hello_world"

    def test_leading_trailing_stripped(self):
        """Leading and trailing whitespace/underscores are stripped."""
        assert sanitize_query_for_path("  hello world  ") == "hello_world"
        assert sanitize_query_for_path("__hello__") == "hello"

    def test_empty_string_returns_default(self):
        """Empty string returns 'query' as fallback."""
        assert sanitize_query_for_path("") == "query"
        assert sanitize_query_for_path("   ") == "query"

    def test_only_special_chars_returns_default(self):
        """String with only special chars returns 'query'."""
        assert sanitize_query_for_path("???") == "query"
        assert sanitize_query_for_path("###") == "query"

    def test_preserves_dots_and_hyphens(self):
        """Dots and hyphens are preserved (useful for filenames)."""
        assert sanitize_query_for_path("file.txt") == "file.txt"
        assert sanitize_query_for_path("my-project") == "my-project"

    def test_unicode_characters(self):
        """Unicode letters are preserved by default (word characters include unicode)."""
        # Python's \w matches unicode letters, so café stays as café
        assert sanitize_query_for_path("café résumé") == "café_résumé"
        # CJK characters are also word characters in Python regex
        assert sanitize_query_for_path("日本語") == "日本語"

    def test_max_length_truncation(self):
        """Long queries are truncated to max_length."""
        long_query = "a" * 200
        result = sanitize_query_for_path(long_query, max_length=120)
        assert len(result) == 120

    def test_max_length_custom(self):
        """Custom max_length is respected."""
        result = sanitize_query_for_path("hello world test", max_length=10)
        assert len(result) <= 10

    def test_emojis_removed(self):
        """Emojis are replaced with underscores."""
        assert sanitize_query_for_path("best 🍕 in america") == "best_in_america"

    def test_quotes_replaced(self):
        """Quotes are replaced with underscores."""
        assert sanitize_query_for_path('"hello world"') == "hello_world"
        assert sanitize_query_for_path("it's working") == "it_s_working"

    def test_newlines_and_tabs(self):
        """Newlines and tabs are treated as whitespace."""
        assert sanitize_query_for_path("hello\nworld") == "hello_world"
        assert sanitize_query_for_path("hello\tworld") == "hello_world"

    def test_realistic_search_queries(self):
        """Test with realistic search query examples."""
        assert sanitize_query_for_path("best pie in america") == "best_pie_in_america"
        assert sanitize_query_for_path("Python 3.12 new features") == "python_3.12_new_features"
        assert sanitize_query_for_path("What is RAG? (Retrieval Augmented Generation)") == "what_is_rag_retrieval_augmented_generation"
        assert sanitize_query_for_path("site:example.com search term") == "site_example.com_search_term"

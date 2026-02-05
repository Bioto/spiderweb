"""Tests for configuration management.

These tests verify that SpiderwebSettings correctly loads from
environment variables and provides sensible defaults.
"""

import os
from unittest.mock import patch

import pytest

from spiderweb.config import SpiderwebSettings, get_settings, reload_settings


class TestSpiderwebSettings:
    """Tests for SpiderwebSettings class."""

    def test_default_values(self):
        """Settings have sensible defaults."""
        settings = SpiderwebSettings()

        # Chunking defaults
        assert settings.default_chunker == "hierarchical"
        assert settings.default_chunk_size == 1000
        assert settings.default_chunk_overlap == 200

        # Vector store defaults
        assert settings.default_vector_store == "memory"
        assert settings.qdrant_host == "localhost"
        assert settings.qdrant_port == 6333

        # Validation defaults
        assert settings.enable_validation is True
        assert settings.min_chunk_quality_score == 0.3
        assert settings.deduplication_threshold == 0.95

        # Search defaults
        assert settings.default_search_provider == "duckduckgo"
        assert settings.default_search_limit == 10

        # Crawler defaults
        assert settings.default_crawler_provider == "crawl4ai"
        assert settings.default_crawler_delay == 1.0
        assert settings.default_crawler_timeout == 30

        # Save format defaults
        assert settings.default_save_format == "all"
        assert settings.default_trace_format == "json"

    def test_env_var_override_chunker(self):
        """Environment variables override chunker settings."""
        with patch.dict(os.environ, {"SPIDERWEB_DEFAULT_CHUNKER": "sentence"}):
            settings = SpiderwebSettings()
            assert settings.default_chunker == "sentence"

    def test_env_var_override_chunk_size(self):
        """Environment variables override chunk size."""
        with patch.dict(os.environ, {"SPIDERWEB_DEFAULT_CHUNK_SIZE": "2000"}):
            settings = SpiderwebSettings()
            assert settings.default_chunk_size == 2000

    def test_env_var_override_search_provider(self):
        """Environment variables override search provider."""
        with patch.dict(os.environ, {"SPIDERWEB_DEFAULT_SEARCH_PROVIDER": "stub"}):
            settings = SpiderwebSettings()
            assert settings.default_search_provider == "stub"

    def test_env_var_override_crawler_settings(self):
        """Environment variables override crawler settings."""
        with patch.dict(
            os.environ,
            {
                "SPIDERWEB_DEFAULT_CRAWLER_PROVIDER": "http",
                "SPIDERWEB_DEFAULT_CRAWLER_DELAY": "2.5",
                "SPIDERWEB_DEFAULT_CRAWLER_TIMEOUT": "60",
            },
        ):
            settings = SpiderwebSettings()
            assert settings.default_crawler_provider == "http"
            assert settings.default_crawler_delay == 2.5
            assert settings.default_crawler_timeout == 60

    def test_env_var_override_search_depth(self):
        """Environment variables override search depth settings."""
        with patch.dict(
            os.environ,
            {
                "SPIDERWEB_DEFAULT_MAX_SEARCH_ROUNDS": "3",
                "SPIDERWEB_DEFAULT_CRAWL_PER_ROUND": "10",
                "SPIDERWEB_DEFAULT_WHEN_TO_GO_DEEPER": "expand_queries",
            },
        ):
            settings = SpiderwebSettings()
            assert settings.default_max_search_rounds == 3
            assert settings.default_crawl_per_round == 10
            assert settings.default_when_to_go_deeper == "expand_queries"

    def test_env_var_override_qdrant(self):
        """Environment variables override Qdrant settings."""
        with patch.dict(
            os.environ,
            {
                "SPIDERWEB_QDRANT_HOST": "qdrant.example.com",
                "SPIDERWEB_QDRANT_PORT": "6334",
                "SPIDERWEB_QDRANT_API_KEY": "secret-key",
            },
        ):
            settings = SpiderwebSettings()
            assert settings.qdrant_host == "qdrant.example.com"
            assert settings.qdrant_port == 6334
            assert settings.qdrant_api_key == "secret-key"

    def test_env_var_case_insensitive(self):
        """Environment variable names are case-insensitive."""
        with patch.dict(os.environ, {"spiderweb_default_chunk_size": "3000"}):
            settings = SpiderwebSettings()
            assert settings.default_chunk_size == 3000

    def test_chunk_size_validation_min(self):
        """Chunk size has minimum validation."""
        with patch.dict(os.environ, {"SPIDERWEB_DEFAULT_CHUNK_SIZE": "50"}):
            with pytest.raises(ValueError):
                SpiderwebSettings()

    def test_chunk_size_validation_max(self):
        """Chunk size has maximum validation."""
        with patch.dict(os.environ, {"SPIDERWEB_DEFAULT_CHUNK_SIZE": "20000"}):
            with pytest.raises(ValueError):
                SpiderwebSettings()

    def test_quality_score_validation(self):
        """Quality score must be between 0 and 1."""
        with patch.dict(os.environ, {"SPIDERWEB_MIN_CHUNK_QUALITY_SCORE": "1.5"}):
            with pytest.raises(ValueError):
                SpiderwebSettings()

    def test_get_log_level(self):
        """get_log_level returns correct logging constants."""
        import logging

        settings = SpiderwebSettings()
        assert settings.get_log_level() == logging.INFO

        with patch.dict(os.environ, {"SPIDERWEB_LOG_LEVEL": "DEBUG"}):
            settings = SpiderwebSettings()
            assert settings.get_log_level() == logging.DEBUG

        with patch.dict(os.environ, {"SPIDERWEB_LOG_LEVEL": "WARNING"}):
            settings = SpiderwebSettings()
            assert settings.get_log_level() == logging.WARNING

    def test_get_file_log_level(self):
        """get_file_log_level returns correct logging constants."""
        import logging

        settings = SpiderwebSettings()
        assert settings.get_file_log_level() == logging.DEBUG

        with patch.dict(os.environ, {"SPIDERWEB_LOG_FILE_LEVEL": "ERROR"}):
            settings = SpiderwebSettings()
            assert settings.get_file_log_level() == logging.ERROR


class TestSettingsCache:
    """Tests for settings caching functionality."""

    def test_get_settings_returns_same_instance(self):
        """get_settings returns cached instance."""
        # Clear cache first
        get_settings.cache_clear()

        s1 = get_settings()
        s2 = get_settings()

        assert s1 is s2

    def test_reload_settings_clears_cache(self):
        """reload_settings creates new instance."""
        # Clear cache first
        get_settings.cache_clear()

        s1 = get_settings()
        s2 = reload_settings()

        # Should be different objects after reload
        assert s1 is not s2

    def test_reload_settings_picks_up_env_changes(self):
        """reload_settings picks up new environment values."""
        get_settings.cache_clear()

        s1 = get_settings()
        original_limit = s1.default_search_limit

        with patch.dict(os.environ, {"SPIDERWEB_DEFAULT_SEARCH_LIMIT": "25"}):
            s2 = reload_settings()
            assert s2.default_search_limit == 25
            assert s2.default_search_limit != original_limit

        # Cleanup
        reload_settings()

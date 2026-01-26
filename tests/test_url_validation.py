"""Tests for crawler URL validation."""

import pytest

from spiderweb.crawlers.url_validation import validate_http_url


def test_validate_http_url_accepts_normal_http_urls():
    assert validate_http_url("https://example.com/path") == "https://example.com/path"
    assert validate_http_url("http://example.com") == "http://example.com"


def test_validate_http_url_rejects_non_http_schemes():
    with pytest.raises(ValueError, match="Unsupported URL scheme"):
        validate_http_url("file:///etc/passwd")

    with pytest.raises(ValueError, match="Unsupported URL scheme"):
        validate_http_url("javascript:alert(1)")


def test_validate_http_url_rejects_embedded_credentials():
    with pytest.raises(ValueError, match="embedded credentials"):
        validate_http_url("http://user:pass@example.com")


def test_validate_http_url_rejects_local_targets():
    with pytest.raises(ValueError, match="localhost"):
        validate_http_url("http://localhost:8000")

    with pytest.raises(ValueError, match="private or non-routable"):
        validate_http_url("http://127.0.0.1")

    with pytest.raises(ValueError, match="private or non-routable"):
        validate_http_url("http://10.0.0.1")



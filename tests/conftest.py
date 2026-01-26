"""Pytest configuration for Spiderweb tests.

Keeps tests isolated by resetting Spiderweb's global settings between tests.
"""

import pytest


@pytest.fixture(autouse=True)
def reset_settings():
    """Reset global settings after each test to ensure test isolation."""
    yield
    # After each test, reload settings to reset to defaults
    from spiderweb.config import reload_settings

    reload_settings()

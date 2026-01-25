"""Observability utilities for Spiderweb.

This module provides logging configuration following gluellm patterns.
"""

from spiderweb.observability.logging_config import get_logger, setup_logging

__all__ = ["setup_logging", "get_logger"]

"""Tests for tracing functionality.

Tests OpenTelemetry tracing enable/disable, span creation, and
attribute/event setting with optional dependency handling.
"""

import sys

import pytest
from unittest.mock import MagicMock, patch

from spiderweb.observability.tracing import (
    add_span_event,
    enable_tracing,
    get_tracer,
    set_span_attribute,
    span,
)


def _make_opentelemetry_mock(mock_trace=None):
    """Build a mock opentelemetry module so 'from opentelemetry import trace' resolves."""
    mock_trace = mock_trace or MagicMock()
    mock_otel = MagicMock()
    mock_otel.trace = mock_trace
    mock_otel.sdk = MagicMock()
    mock_otel.sdk.trace = MagicMock()
    mock_otel.sdk.trace.TracerProvider = MagicMock()
    mock_otel.sdk.trace.export = MagicMock()
    mock_otel.sdk.trace.export.BatchSpanProcessor = MagicMock()
    mock_otel.sdk.trace.export.ConsoleSpanExporter = MagicMock()
    return mock_otel


@pytest.fixture(autouse=True)
def reset_tracing():
    """Reset tracing state before and after each test."""
    enable_tracing(False)
    yield
    enable_tracing(False)


class TestTracingDisabled:
    """Tests for tracing when disabled."""

    def test_get_tracer_returns_none_when_disabled(self):
        """get_tracer returns None when tracing disabled."""
        enable_tracing(False)
        assert get_tracer() is None

    def test_span_yields_none_when_disabled(self):
        """span context manager yields None when disabled."""
        enable_tracing(False)
        with span("test_span") as span_obj:
            assert span_obj is None

    def test_set_span_attribute_no_exception_when_disabled(self):
        """set_span_attribute doesn't raise when disabled."""
        enable_tracing(False)
        # Should not raise
        set_span_attribute("key", "value")

    def test_add_span_event_no_exception_when_disabled(self):
        """add_span_event doesn't raise when disabled."""
        enable_tracing(False)
        # Should not raise
        add_span_event("event_name")


class TestTracingEnabled:
    """Tests for tracing when enabled."""

    def test_enable_tracing_without_opentelemetry(self):
        """enable_tracing handles missing opentelemetry gracefully."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "opentelemetry":
                raise ImportError("No module named 'opentelemetry'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            enable_tracing(True)
        assert get_tracer() is None

    def test_enable_tracing_with_opentelemetry(self):
        """enable_tracing creates tracer when opentelemetry available."""
        mock_tracer = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer = MagicMock(return_value=mock_tracer)
        mock_trace.set_tracer_provider = MagicMock()
        mock_otel = _make_opentelemetry_mock(mock_trace)
        # Sub-imports in enable_tracing need these in sys.modules
        otel_modules = {
            "opentelemetry": mock_otel,
            "opentelemetry.sdk": mock_otel.sdk,
            "opentelemetry.sdk.trace": mock_otel.sdk.trace,
            "opentelemetry.sdk.trace.export": mock_otel.sdk.trace.export,
        }

        with patch.dict(sys.modules, otel_modules):
            enable_tracing(True)
        tracer = get_tracer()
        assert tracer is not None

    def test_span_creates_span_when_enabled(self):
        """span creates span when tracing enabled."""
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_context_manager = MagicMock()
        mock_context_manager.__enter__ = MagicMock(return_value=mock_span)
        mock_context_manager.__exit__ = MagicMock(return_value=None)
        mock_tracer.start_as_current_span = MagicMock(return_value=mock_context_manager)

        with patch("spiderweb.observability.tracing.get_tracer", return_value=mock_tracer):
            with span("test_span") as span_obj:
                assert span_obj is not None
                mock_tracer.start_as_current_span.assert_called_once_with("test_span")

    def test_span_sets_attributes(self):
        """span sets attributes on span when provided."""
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_context_manager = MagicMock()
        mock_context_manager.__enter__ = MagicMock(return_value=mock_span)
        mock_context_manager.__exit__ = MagicMock(return_value=None)
        mock_tracer.start_as_current_span = MagicMock(return_value=mock_context_manager)

        with patch("spiderweb.observability.tracing.get_tracer", return_value=mock_tracer):
            with span("test_span", attributes={"key1": "value1", "key2": 42}):
                # Should set attributes
                assert mock_span.set_attribute.call_count == 2
                mock_span.set_attribute.assert_any_call("key1", "value1")
                mock_span.set_attribute.assert_any_call("key2", "42")  # Converted to string

    def test_set_span_attribute_when_enabled(self):
        """set_span_attribute sets attribute when tracing enabled."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        mock_trace = MagicMock()
        mock_trace.get_current_span = MagicMock(return_value=mock_span)
        mock_otel = _make_opentelemetry_mock(mock_trace)

        with patch("spiderweb.observability.tracing._enabled", True), \
             patch.dict(sys.modules, {"opentelemetry": mock_otel}):
            set_span_attribute("test_key", "test_value")
        mock_span.set_attribute.assert_called_once_with("test_key", "test_value")

    def test_set_span_attribute_no_current_span(self):
        """set_span_attribute handles no current span gracefully."""
        mock_trace = MagicMock()
        mock_trace.get_current_span = MagicMock(return_value=None)
        mock_otel = _make_opentelemetry_mock(mock_trace)

        with patch("spiderweb.observability.tracing._enabled", True), \
             patch.dict(sys.modules, {"opentelemetry": mock_otel}):
            set_span_attribute("key", "value")

    def test_set_span_attribute_span_not_recording(self):
        """set_span_attribute skips when span not recording."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = False
        mock_trace = MagicMock()
        mock_trace.get_current_span = MagicMock(return_value=mock_span)
        mock_otel = _make_opentelemetry_mock(mock_trace)

        with patch("spiderweb.observability.tracing._enabled", True), \
             patch.dict(sys.modules, {"opentelemetry": mock_otel}):
            set_span_attribute("key", "value")
        mock_span.set_attribute.assert_not_called()

    def test_add_span_event_when_enabled(self):
        """add_span_event adds event when tracing enabled."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        mock_trace = MagicMock()
        mock_trace.get_current_span = MagicMock(return_value=mock_span)
        mock_otel = _make_opentelemetry_mock(mock_trace)

        with patch("spiderweb.observability.tracing._enabled", True), \
             patch.dict(sys.modules, {"opentelemetry": mock_otel}):
            add_span_event("test_event", attributes={"key": "value"})
        mock_span.add_event.assert_called_once_with("test_event", attributes={"key": "value"})

    def test_add_span_event_no_attributes(self):
        """add_span_event works without attributes."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        mock_trace = MagicMock()
        mock_trace.get_current_span = MagicMock(return_value=mock_span)
        mock_otel = _make_opentelemetry_mock(mock_trace)

        with patch("spiderweb.observability.tracing._enabled", True), \
             patch.dict(sys.modules, {"opentelemetry": mock_otel}):
            add_span_event("test_event")
        mock_span.add_event.assert_called_once_with("test_event", attributes={})

    def test_set_span_attribute_exception_handling(self):
        """set_span_attribute handles exceptions gracefully."""
        mock_trace = MagicMock()
        mock_trace.get_current_span = MagicMock(side_effect=Exception("Error"))
        mock_otel = _make_opentelemetry_mock(mock_trace)

        with patch("spiderweb.observability.tracing._enabled", True), \
             patch.dict(sys.modules, {"opentelemetry": mock_otel}):
            set_span_attribute("key", "value")

    def test_add_span_event_exception_handling(self):
        """add_span_event handles exceptions gracefully."""
        mock_trace = MagicMock()
        mock_trace.get_current_span = MagicMock(side_effect=Exception("Error"))
        mock_otel = _make_opentelemetry_mock(mock_trace)

        with patch("spiderweb.observability.tracing._enabled", True), \
             patch.dict(sys.modules, {"opentelemetry": mock_otel}):
            add_span_event("event_name")


class TestTracingToggle:
    """Tests for toggling tracing on/off."""

    def test_enable_then_disable(self):
        """Can enable then disable tracing."""
        enable_tracing(True)
        # May or may not have tracer depending on opentelemetry availability
        enable_tracing(False)
        assert get_tracer() is None

    def test_multiple_enable_calls(self):
        """Multiple enable calls are idempotent."""
        enable_tracing(True)
        tracer1 = get_tracer()
        enable_tracing(True)
        tracer2 = get_tracer()
        # Both should be same (or both None if opentelemetry not available)
        assert (tracer1 is None) == (tracer2 is None)

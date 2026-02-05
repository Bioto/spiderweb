"""OpenTelemetry tracing for Spiderweb operations.

Provides distributed tracing spans for key operations including ingestion,
querying, and crawling. Tracing is optional and can be disabled.
"""

from contextlib import contextmanager
from typing import Any, Generator

from spiderweb.observability.logging_config import get_logger

logger = get_logger(__name__)

# Global tracer (None = disabled)
_tracer = None
_enabled = False


def enable_tracing(enabled: bool = True) -> None:
    """Enable or disable tracing.

    Args:
        enabled: Whether to enable tracing
    """
    global _enabled, _tracer

    _enabled = enabled

    if enabled:
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter, BatchSpanProcessor

            # Initialize tracer provider
            provider = TracerProvider()
            processor = BatchSpanProcessor(ConsoleSpanExporter())
            provider.add_span_processor(processor)
            trace.set_tracer_provider(provider)

            _tracer = trace.get_tracer("spiderweb")
            logger.info("OpenTelemetry tracing enabled")
        except ImportError:
            logger.warning(
                "OpenTelemetry not installed. Install with: pip install opentelemetry-api opentelemetry-sdk. "
                "Tracing disabled."
            )
            _enabled = False
            _tracer = None
    else:
        _tracer = None
        logger.info("Tracing disabled")


def get_tracer() -> Any:
    """Get the global tracer instance.

    Returns:
        Tracer instance or None if disabled
    """
    return _tracer if _enabled else None


@contextmanager
def span(
    name: str,
    attributes: dict[str, Any] | None = None,
) -> Generator[Any, None, None]:
    """Create a tracing span context manager.

    Args:
        name: Span name
        attributes: Optional span attributes

    Yields:
        Span object

    Example:
        >>> with span("ingest_document", {"file": "doc.pdf"}):
        ...     # Do work
        ...     pass
    """
    tracer = get_tracer()
    if not tracer:
        yield None
        return

    with tracer.start_as_current_span(name) as span_obj:
        if attributes and span_obj:
            for key, value in attributes.items():
                span_obj.set_attribute(key, str(value))
        yield span_obj


def set_span_attribute(key: str, value: Any) -> None:
    """Set an attribute on the current span.

    Args:
        key: Attribute key
        value: Attribute value
    """
    if not _enabled:
        return

    try:
        from opentelemetry import trace

        current_span = trace.get_current_span()
        if current_span and current_span.is_recording():
            current_span.set_attribute(key, str(value))
    except Exception:
        # Tracing is optional, don't fail if it's not available
        pass


def add_span_event(name: str, attributes: dict[str, Any] | None = None) -> None:
    """Add an event to the current span.

    Args:
        name: Event name
        attributes: Optional event attributes
    """
    if not _enabled:
        return

    try:
        from opentelemetry import trace

        current_span = trace.get_current_span()
        if current_span and current_span.is_recording():
            current_span.add_event(name, attributes=attributes or {})
    except Exception:
        pass

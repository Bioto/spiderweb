"""Metrics collection for Spiderweb operations.

Provides Prometheus-style metrics for key operations including ingestion,
querying, and crawling. Metrics are optional and can be disabled.
"""

from typing import Any

from spiderweb.observability.logging_config import get_logger

logger = get_logger(__name__)

# Global metrics registry (simple dict-based, can be replaced with prometheus_client)
_metrics_registry: dict[str, Any] = {
    "counters": {},
    "histograms": {},
    "gauges": {},
}


class MetricsCollector:
    """Simple metrics collector for Spiderweb operations.

    Tracks counters, histograms, and gauges for observability.
    Can be extended to export to Prometheus or OpenTelemetry.
    """

    def __init__(self, enabled: bool = True):
        """Initialize metrics collector.

        Args:
            enabled: Whether to collect metrics (can be disabled for performance)
        """
        self.enabled = enabled

    def increment_counter(
        self,
        name: str,
        value: float = 1.0,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Increment a counter metric.

        Args:
            name: Metric name (e.g., "spiderweb_ingest_documents_total")
            value: Value to increment by
            labels: Optional labels/tags
        """
        if not self.enabled:
            return

        key = self._make_key(name, labels)
        _metrics_registry["counters"][key] = _metrics_registry["counters"].get(key, 0.0) + value

    def record_histogram(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Record a histogram value.

        Args:
            name: Metric name (e.g., "spiderweb_query_duration_seconds")
            value: Value to record
            labels: Optional labels/tags
        """
        if not self.enabled:
            return

        key = self._make_key(name, labels)
        if key not in _metrics_registry["histograms"]:
            _metrics_registry["histograms"][key] = []
        _metrics_registry["histograms"][key].append(value)

    def set_gauge(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Set a gauge value.

        Args:
            name: Metric name (e.g., "spiderweb_chunks_stored")
            value: Gauge value
            labels: Optional labels/tags
        """
        if not self.enabled:
            return

        key = self._make_key(name, labels)
        _metrics_registry["gauges"][key] = value

    def _make_key(self, name: str, labels: dict[str, str] | None) -> str:
        """Create a key from metric name and labels.

        Args:
            name: Metric name
            labels: Optional labels

        Returns:
            Composite key string
        """
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def get_metrics(self) -> dict[str, Any]:
        """Get all collected metrics.

        Returns:
            Dict with counters, histograms, and gauges
        """
        return {
            "counters": dict(_metrics_registry["counters"]),
            "histograms": {
                k: {
                    "count": len(v),
                    "sum": sum(v),
                    "min": min(v) if v else 0,
                    "max": max(v) if v else 0,
                    "avg": sum(v) / len(v) if v else 0,
                }
                for k, v in _metrics_registry["histograms"].items()
            },
            "gauges": dict(_metrics_registry["gauges"]),
        }

    def reset(self) -> None:
        """Reset all metrics."""
        _metrics_registry["counters"].clear()
        _metrics_registry["histograms"].clear()
        _metrics_registry["gauges"].clear()


# Global metrics collector instance
_global_collector = MetricsCollector(enabled=False)  # Disabled by default


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector.

    Returns:
        Global MetricsCollector instance
    """
    return _global_collector


def enable_metrics(enabled: bool = True) -> None:
    """Enable or disable metrics collection.

    Args:
        enabled: Whether to enable metrics
    """
    _global_collector.enabled = enabled
    logger.info(f"Metrics collection {'enabled' if enabled else 'disabled'}")


# Convenience functions for common metrics
def record_ingest_duration(seconds: float, success: bool = True) -> None:
    """Record document ingestion duration.

    Args:
        seconds: Processing time in seconds
        success: Whether ingestion succeeded
    """
    _global_collector.record_histogram(
        "spiderweb_ingest_duration_seconds",
        seconds,
        labels={"status": "success" if success else "failure"},
    )


def record_query_duration(seconds: float) -> None:
    """Record query duration.

    Args:
        seconds: Query time in seconds
    """
    _global_collector.record_histogram("spiderweb_query_duration_seconds", seconds)


def increment_chunks_created(count: int = 1) -> None:
    """Increment chunks created counter.

    Args:
        count: Number of chunks created
    """
    _global_collector.increment_counter("spiderweb_chunks_created_total", value=float(count))


def increment_documents_processed(count: int = 1, success: bool = True) -> None:
    """Increment documents processed counter.

    Args:
        count: Number of documents
        success: Whether processing succeeded
    """
    _global_collector.increment_counter(
        "spiderweb_documents_processed_total",
        value=float(count),
        labels={"status": "success" if success else "failure"},
    )

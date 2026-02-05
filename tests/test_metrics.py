"""Tests for metrics collection functionality.

Tests MetricsCollector behavior, enabled/disabled state, persistence,
and convenience functions.
"""

import pytest

from spiderweb.observability.metrics import (
    MetricsCollector,
    enable_metrics,
    get_metrics_collector,
    increment_chunks_created,
    increment_documents_processed,
    record_ingest_duration,
    record_query_duration,
)


@pytest.fixture(autouse=True)
def reset_metrics():
    """Reset metrics registry before and after each test."""
    collector = MetricsCollector(enabled=True)
    collector.reset()
    yield
    collector.reset()


class TestMetricsCollectorEnabled:
    """Tests for MetricsCollector with enabled=True."""

    def test_increment_counter(self):
        """increment_counter increases counter value."""
        collector = MetricsCollector(enabled=True)
        collector.increment_counter("test_counter")
        
        metrics = collector.get_metrics()
        assert metrics["counters"]["test_counter"] == 1.0

    def test_increment_counter_with_value(self):
        """increment_counter accepts custom increment value."""
        collector = MetricsCollector(enabled=True)
        collector.increment_counter("test_counter", value=5.0)
        
        metrics = collector.get_metrics()
        assert metrics["counters"]["test_counter"] == 5.0

    def test_increment_counter_multiple_times(self):
        """Multiple increments accumulate."""
        collector = MetricsCollector(enabled=True)
        collector.increment_counter("test_counter", value=2.0)
        collector.increment_counter("test_counter", value=3.0)
        
        metrics = collector.get_metrics()
        assert metrics["counters"]["test_counter"] == 5.0

    def test_increment_counter_with_labels(self):
        """Counters with different labels are separate."""
        collector = MetricsCollector(enabled=True)
        collector.increment_counter("test_counter", labels={"status": "success"})
        collector.increment_counter("test_counter", labels={"status": "failure"})
        
        metrics = collector.get_metrics()
        assert metrics["counters"]["test_counter{status=failure}"] == 1.0
        assert metrics["counters"]["test_counter{status=success}"] == 1.0

    def test_record_histogram(self):
        """record_histogram stores values."""
        collector = MetricsCollector(enabled=True)
        collector.record_histogram("test_histogram", 1.5)
        collector.record_histogram("test_histogram", 2.5)
        collector.record_histogram("test_histogram", 3.5)
        
        metrics = collector.get_metrics()
        hist = metrics["histograms"]["test_histogram"]
        assert hist["count"] == 3
        assert hist["sum"] == 7.5
        assert hist["min"] == 1.5
        assert hist["max"] == 3.5
        assert hist["avg"] == 2.5

    def test_record_histogram_with_labels(self):
        """Histograms with different labels are separate."""
        collector = MetricsCollector(enabled=True)
        collector.record_histogram("test_hist", 1.0, labels={"type": "a"})
        collector.record_histogram("test_hist", 2.0, labels={"type": "b"})
        
        metrics = collector.get_metrics()
        assert metrics["histograms"]["test_hist{type=a}"]["sum"] == 1.0
        assert metrics["histograms"]["test_hist{type=b}"]["sum"] == 2.0

    def test_set_gauge(self):
        """set_gauge sets gauge value."""
        collector = MetricsCollector(enabled=True)
        collector.set_gauge("test_gauge", 42.0)
        
        metrics = collector.get_metrics()
        assert metrics["gauges"]["test_gauge"] == 42.0

    def test_set_gauge_overwrites(self):
        """set_gauge overwrites previous value."""
        collector = MetricsCollector(enabled=True)
        collector.set_gauge("test_gauge", 10.0)
        collector.set_gauge("test_gauge", 20.0)
        
        metrics = collector.get_metrics()
        assert metrics["gauges"]["test_gauge"] == 20.0

    def test_set_gauge_with_labels(self):
        """Gauges with different labels are separate."""
        collector = MetricsCollector(enabled=True)
        collector.set_gauge("test_gauge", 1.0, labels={"env": "prod"})
        collector.set_gauge("test_gauge", 2.0, labels={"env": "dev"})
        
        metrics = collector.get_metrics()
        assert metrics["gauges"]["test_gauge{env=dev}"] == 2.0
        assert metrics["gauges"]["test_gauge{env=prod}"] == 1.0


class TestMetricsCollectorDisabled:
    """Tests for MetricsCollector with enabled=False."""

    def test_increment_counter_disabled(self):
        """increment_counter does nothing when disabled."""
        collector = MetricsCollector(enabled=False)
        collector.increment_counter("test_counter")
        
        metrics = collector.get_metrics()
        assert "test_counter" not in metrics["counters"]

    def test_record_histogram_disabled(self):
        """record_histogram does nothing when disabled."""
        collector = MetricsCollector(enabled=False)
        collector.record_histogram("test_histogram", 1.5)
        
        metrics = collector.get_metrics()
        assert "test_histogram" not in metrics["histograms"]

    def test_set_gauge_disabled(self):
        """set_gauge does nothing when disabled."""
        collector = MetricsCollector(enabled=False)
        collector.set_gauge("test_gauge", 42.0)
        
        metrics = collector.get_metrics()
        assert "test_gauge" not in metrics["gauges"]


class TestMetricsCollectorReset:
    """Tests for reset method."""

    def test_reset_clears_all_metrics(self):
        """reset clears all counters, histograms, and gauges."""
        collector = MetricsCollector(enabled=True)
        collector.increment_counter("counter1")
        collector.record_histogram("hist1", 1.0)
        collector.set_gauge("gauge1", 10.0)
        
        collector.reset()
        
        metrics = collector.get_metrics()
        assert len(metrics["counters"]) == 0
        assert len(metrics["histograms"]) == 0
        assert len(metrics["gauges"]) == 0


class TestConvenienceFunctions:
    """Tests for convenience metric functions."""

    def test_record_ingest_duration_success(self):
        """record_ingest_duration records histogram with success label."""
        enable_metrics(True)
        get_metrics_collector().reset()
        
        record_ingest_duration(1.5, success=True)
        
        collector = get_metrics_collector()
        metrics = collector.get_metrics()
        hist = metrics["histograms"]["spiderweb_ingest_duration_seconds{status=success}"]
        assert hist["count"] == 1
        assert hist["sum"] == 1.5

    def test_record_ingest_duration_failure(self):
        """record_ingest_duration records histogram with failure label."""
        enable_metrics(True)
        get_metrics_collector().reset()
        
        record_ingest_duration(2.0, success=False)
        
        collector = get_metrics_collector()
        metrics = collector.get_metrics()
        hist = metrics["histograms"]["spiderweb_ingest_duration_seconds{status=failure}"]
        assert hist["count"] == 1
        assert hist["sum"] == 2.0

    def test_record_query_duration(self):
        """record_query_duration records histogram."""
        enable_metrics(True)
        get_metrics_collector().reset()
        
        record_query_duration(0.5)
        
        collector = get_metrics_collector()
        metrics = collector.get_metrics()
        hist = metrics["histograms"]["spiderweb_query_duration_seconds"]
        assert hist["count"] == 1
        assert hist["sum"] == 0.5

    def test_increment_chunks_created(self):
        """increment_chunks_created increments counter."""
        enable_metrics(True)
        get_metrics_collector().reset()
        
        increment_chunks_created(count=5)
        
        collector = get_metrics_collector()
        metrics = collector.get_metrics()
        assert metrics["counters"]["spiderweb_chunks_created_total"] == 5.0

    def test_increment_chunks_created_default(self):
        """increment_chunks_created defaults to count=1."""
        enable_metrics(True)
        get_metrics_collector().reset()
        
        increment_chunks_created()
        
        collector = get_metrics_collector()
        metrics = collector.get_metrics()
        assert metrics["counters"]["spiderweb_chunks_created_total"] == 1.0

    def test_increment_documents_processed_success(self):
        """increment_documents_processed increments counter with success label."""
        enable_metrics(True)
        get_metrics_collector().reset()
        
        increment_documents_processed(count=3, success=True)
        
        collector = get_metrics_collector()
        metrics = collector.get_metrics()
        assert metrics["counters"]["spiderweb_documents_processed_total{status=success}"] == 3.0

    def test_increment_documents_processed_failure(self):
        """increment_documents_processed increments counter with failure label."""
        enable_metrics(True)
        get_metrics_collector().reset()
        
        increment_documents_processed(count=2, success=False)
        
        collector = get_metrics_collector()
        metrics = collector.get_metrics()
        assert metrics["counters"]["spiderweb_documents_processed_total{status=failure}"] == 2.0

    def test_convenience_functions_when_disabled(self):
        """Convenience functions don't crash when metrics disabled."""
        enable_metrics(False)
        get_metrics_collector().reset()
        
        # Should not raise
        record_ingest_duration(1.0)
        record_query_duration(0.5)
        increment_chunks_created()
        increment_documents_processed()
        
        collector = get_metrics_collector()
        metrics = collector.get_metrics()
        assert len(metrics["counters"]) == 0
        assert len(metrics["histograms"]) == 0


class TestMetricsIsolation:
    """Tests for metrics isolation."""

    def test_multiple_collectors_share_registry(self):
        """Multiple collectors share the same global registry."""
        collector1 = MetricsCollector(enabled=True)
        collector2 = MetricsCollector(enabled=True)
        
        collector1.increment_counter("shared_counter")
        
        metrics1 = collector1.get_metrics()
        metrics2 = collector2.get_metrics()
        
        assert metrics1["counters"]["shared_counter"] == 1.0
        assert metrics2["counters"]["shared_counter"] == 1.0

    def test_global_collector_via_get_metrics_collector(self):
        """get_metrics_collector returns the global collector."""
        collector1 = get_metrics_collector()
        collector2 = get_metrics_collector()
        
        assert collector1 is collector2

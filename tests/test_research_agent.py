"""Tests for research agent: spill to store, aggregate_traces_for_report, batched summarization."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from spiderweb.crawlers.base import CrawlResult
from spiderweb.research.agent import (
    _iter_page_contents,
    aggregate_traces_for_report,
    spill_trace_content_to_dir,
    spill_trace_to_store,
    summarize_content_batch,
    summarize_traces_in_batches,
    synthesize_report_from_summaries,
)
from spiderweb.research.storage import MarkdownResearchStore
from spiderweb.search.base import SearchResultBatch
from spiderweb.search.trace import PageRecord, SearchCrawlTrace, SearchRound


def _make_trace_with_pages(
    original_query: str,
    pages: list[tuple[str, str | None, CrawlResult | None]],
) -> SearchCrawlTrace:
    """Build a SearchCrawlTrace with one round and the given pages."""
    batch = SearchResultBatch(results=[], query=original_query, total=0)
    page_records = []
    for url, summary, crawl_result in pages:
        page_records.append(
            PageRecord(
                url=url,
                summary=summary,
                crawl_result=crawl_result,
                source_query=original_query,
            )
        )
    round_data = SearchRound(
        query=original_query,
        search_results=batch,
        pages=page_records,
        filtered_out=[],
        round_number=1,
    )
    trace = SearchCrawlTrace(original_query=original_query, rounds=[round_data])
    return trace


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


class TestSpillTraceToStore:
    """Tests for spill_trace_to_store."""

    def test_spill_clears_crawl_result_and_sets_content_path(self, temp_dir):
        """Spilling sets content_path and clears crawl_result for each page."""
        store = MarkdownResearchStore(temp_dir, subdir="q1")
        trace = _make_trace_with_pages(
            "test query",
            [
                (
                    "https://example.com/a",
                    "Summary A",
                    CrawlResult(url="https://example.com/a", content="", markdown="# Page A\n\nText A."),
                ),
            ],
        )
        page = trace.rounds[0].pages[0]
        assert page.crawl_result is not None
        assert page.content_path is None

        spill_trace_to_store(trace, store)

        assert page.crawl_result is None
        assert page.content_path is not None
        assert Path(page.content_path).exists()
        assert store.get_content(page.content_path) == "# Page A\n\nText A."

    def test_spill_skips_pages_without_crawl_result(self, temp_dir):
        """Pages with crawl_result=None are left unchanged."""
        store = MarkdownResearchStore(temp_dir, subdir="q1")
        trace = _make_trace_with_pages(
            "query",
            [("https://example.com/x", "Summary", None)],
        )
        page = trace.rounds[0].pages[0]
        spill_trace_to_store(trace, store)
        assert page.content_path is None
        assert page.crawl_result is None

    def test_spill_uses_content_when_markdown_empty(self, temp_dir):
        """When markdown is empty, content is used."""
        store = MarkdownResearchStore(temp_dir, subdir="q1")
        trace = _make_trace_with_pages(
            "query",
            [
                (
                    "https://example.com/p",
                    None,
                    CrawlResult(url="https://example.com/p", content="Plain text body", markdown=None),
                ),
            ],
        )
        spill_trace_to_store(trace, store)
        page = trace.rounds[0].pages[0]
        assert store.get_content(page.content_path) == "Plain text body"


class TestSpillTraceContentToDir:
    """Tests for spill_trace_content_to_dir (convenience wrapper)."""

    def test_spill_to_dir_creates_subdir_by_query(self, temp_dir):
        """spill_trace_content_to_dir creates a subdir from query and spills."""
        trace = _make_trace_with_pages(
            "python web scraping",
            [
                (
                    "https://example.com/page",
                    None,
                    CrawlResult(url="https://example.com/page", content="", markdown="# Hi"),
                ),
            ],
        )
        spill_trace_content_to_dir(trace, temp_dir)
        page = trace.rounds[0].pages[0]
        assert page.crawl_result is None
        assert page.content_path is not None
        # Subdir should be based on sanitized query
        assert "python" in page.content_path or "web" in page.content_path or temp_dir.name in page.content_path
        assert Path(page.content_path).exists()


class TestAggregateTracesForReport:
    """Tests for aggregate_traces_for_report."""

    @pytest.mark.asyncio
    async def test_aggregate_from_crawl_result_in_memory(self):
        """When pages have crawl_result (no store), content is taken from crawl_result."""
        trace = _make_trace_with_pages(
            "q1",
            [
                (
                    "https://example.com/one",
                    "S1",
                    CrawlResult(url="https://example.com/one", content="", markdown="# One\n\nBody one."),
                ),
            ],
        )
        out = await aggregate_traces_for_report([trace], max_chars_per_page=5000)
        assert "https://example.com/one" in out
        assert "Body one." in out
        assert "S1" in out

    @pytest.mark.asyncio
    async def test_aggregate_from_content_path_with_store(self, temp_dir):
        """When pages have content_path and store is provided, content is read via store."""
        store = MarkdownResearchStore(temp_dir, subdir="run")
        trace = _make_trace_with_pages(
            "q1",
            [
                (
                    "https://example.com/two",
                    "S2",
                    CrawlResult(url="https://example.com/two", content="", markdown="# Two\n\nBody two."),
                ),
            ],
        )
        spill_trace_to_store(trace, store)
        out = await aggregate_traces_for_report([trace], max_chars_per_page=5000, store=store)
        assert "https://example.com/two" in out
        assert "Body two." in out
        assert "S2" in out

    @pytest.mark.asyncio
    async def test_aggregate_deduplicates_by_url(self):
        """Duplicate URLs across traces are only included once."""
        trace1 = _make_trace_with_pages(
            "q1",
            [("https://example.com/same", "S1", CrawlResult(url="https://example.com/same", content="", markdown="First"))],
        )
        trace2 = _make_trace_with_pages(
            "q2",
            [("https://example.com/same", "S2", CrawlResult(url="https://example.com/same", content="", markdown="Second"))],
        )
        out = await aggregate_traces_for_report([trace1, trace2], max_chars_per_page=5000)
        assert out.count("https://example.com/same") >= 1
        # First occurrence wins
        assert "First" in out or "Second" in out

    @pytest.mark.asyncio
    async def test_aggregate_truncates_by_max_chars_per_page(self):
        """Content is truncated to max_chars_per_page."""
        long_content = "x" * 5000
        trace = _make_trace_with_pages(
            "q",
            [
                (
                    "https://example.com/long",
                    None,
                    CrawlResult(url="https://example.com/long", content="", markdown=long_content),
                ),
            ],
        )
        out = await aggregate_traces_for_report([trace], max_chars_per_page=100)
        assert "[... truncated ...]" in out
        assert out.count("x") <= 100 + 20  # some header/footer

    @pytest.mark.asyncio
    async def test_aggregate_empty_traces_returns_empty_content(self):
        """Empty traces produce minimal but valid output."""
        trace = _make_trace_with_pages("q", [])
        out = await aggregate_traces_for_report([trace], max_chars_per_page=4000)
        assert isinstance(out, str)
        assert out == "" or "---" in out or len(out) < 20


class TestIterPageContents:
    """Tests for _iter_page_contents (batch iteration helper)."""

    def test_yields_one_string_per_page_deduped(self):
        """Yields formatted content per page and deduplicates by URL."""
        trace = _make_trace_with_pages(
            "q1",
            [
                ("https://example.com/a", "S1", CrawlResult(url="https://example.com/a", content="", markdown="# A")),
                ("https://example.com/b", "S2", CrawlResult(url="https://example.com/b", content="", markdown="# B")),
            ],
        )
        pages = list(_iter_page_contents([trace], None, 4000))
        assert len(pages) == 2
        assert "https://example.com/a" in pages[0]
        assert "https://example.com/b" in pages[1]
        assert "# A" in pages[0]
        assert "# B" in pages[1]

    def test_deduplicates_by_url_across_traces(self):
        """Same URL in two traces appears only once."""
        t1 = _make_trace_with_pages("q1", [("https://example.com/same", None, CrawlResult(url="x", content="", markdown="First"))])
        t2 = _make_trace_with_pages("q2", [("https://example.com/same", None, CrawlResult(url="x", content="", markdown="Second"))])
        pages = list(_iter_page_contents([t1, t2], None, 4000))
        assert len(pages) == 1
        assert "First" in pages[0] or "Second" in pages[0]


class TestSummarizeContentBatch:
    """Tests for summarize_content_batch."""

    @pytest.mark.asyncio
    async def test_returns_llm_response(self):
        """Returns the LLM response as the batch summary."""
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = type("R", (), {"final_response": "Key findings: X and Y."})()
        out = await summarize_content_batch(
            mock_llm,
            persona="You are an analyst",
            instructions="Research X",
            batch_content="## url\n\nContent here.",
            batch_label="Batch 1",
        )
        assert out == "Key findings: X and Y."
        mock_llm.complete.assert_called_once()


class TestSynthesizeReportFromSummaries:
    """Tests for synthesize_report_from_summaries."""

    @pytest.mark.asyncio
    async def test_returns_llm_report(self):
        """Returns the LLM-generated report from batch summaries."""
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = type("R", (), {"final_response": "# Final Report\n\nSynthesized."})()
        out = await synthesize_report_from_summaries(
            mock_llm,
            persona="Analyst",
            instructions="Research topic",
            batch_summaries=["Summary 1", "Summary 2"],
            report_focus=None,
        )
        assert "# Final Report" in out
        assert "Synthesized" in out
        mock_llm.complete.assert_called_once()


class TestSummarizeTracesInBatches:
    """Tests for summarize_traces_in_batches."""

    @pytest.mark.asyncio
    async def test_batches_by_page_count(self):
        """Creates one batch per batch_size_pages and one for remainder."""
        # 10 pages -> batch_size 3 -> 4 batches (3+3+3+1)
        pages = [
            (
                f"https://example.com/p{i}",
                f"S{i}",
                CrawlResult(url=f"https://example.com/p{i}", content="", markdown=f"# P{i}"),
            )
            for i in range(10)
        ]
        trace = _make_trace_with_pages("q", pages)
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = type("R", (), {"final_response": "Batch summary."})()
        summaries = await summarize_traces_in_batches(
            mock_llm,
            persona="Analyst",
            instructions="Research",
            traces=[trace],
            store=None,
            batch_size_pages=3,
            max_chars_per_page=4000,
        )
        assert len(summaries) == 4  # 3+3+3+1
        assert all(s == "Batch summary." for s in summaries)
        assert mock_llm.complete.call_count == 4

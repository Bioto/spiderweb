"""Writers for SearchCrawlTrace in various formats.

Supports JSON, markdown, and JSONL output for different use cases:
- JSON: single file, full trace
- Markdown: human-readable summary
- JSONL: streaming/processing, one record per line
"""

import json
from pathlib import Path
from typing import Literal

from spiderweb.search.trace import FilteredCandidate, PageRecord, SearchCrawlTrace, SearchRound


class TraceWriter:
    """Base class for trace writers."""
    
    def write(self, trace: SearchCrawlTrace, path: Path) -> None:
        """Write trace to file.
        
        Args:
            trace: SearchCrawlTrace to write
            path: Output file path
        """
        raise NotImplementedError


class JSONTraceWriter(TraceWriter):
    """Write trace as JSON (single file, full structure)."""
    
    def write(self, trace: SearchCrawlTrace, path: Path) -> None:
        """Write trace as JSON.
        
        Args:
            trace: SearchCrawlTrace to write
            path: Output JSON file path
        """
        data = self._trace_to_dict(trace)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _trace_to_dict(self, trace: SearchCrawlTrace) -> dict:
        """Convert trace to dictionary for JSON serialization."""
        return {
            "original_query": trace.original_query,
            "rounds": [self._round_to_dict(r) for r in trace.rounds],
            "config_snapshot": trace.config_snapshot,
        }
    
    def _round_to_dict(self, round_data: SearchRound) -> dict:
        """Convert SearchRound to dictionary."""
        return {
            "round_number": round_data.round_number,
            "query": round_data.query,
            "search_results": {
                "query": round_data.search_results.query,
                "total": round_data.search_results.total,
                "results": [
                    {
                        "url": r.url,
                        "title": r.title,
                        "description": r.description,
                        "position": r.position,
                        "source": r.source,
                        "category": r.category,
                    }
                    for r in round_data.search_results.results
                ],
            },
            "pages": [self._page_to_dict(p) for p in round_data.pages],
            "filtered_out": [self._filtered_to_dict(f) for f in round_data.filtered_out],
        }
    
    def _page_to_dict(self, page: PageRecord) -> dict:
        """Convert PageRecord to dictionary."""
        return {
            "url": page.url,
            "summary": page.summary,
            "source_query": page.source_query,
            "source_position": page.source_position,
            "parent_url": page.parent_url,
            "links_found": page.links_found,
            "extracted_data": page.extracted_data,
            "crawl_status": "success" if page.crawl_result and page.crawl_result.success else "failed",
            "status_code": page.crawl_result.status_code if page.crawl_result else None,
        }
    
    def _filtered_to_dict(self, filtered: FilteredCandidate) -> dict:
        """Convert FilteredCandidate to dictionary."""
        return {
            "url": filtered.url,
            "title": filtered.title,
            "snippet": filtered.snippet,
            "source_query": filtered.source_query,
            "source_position": filtered.source_position,
            "parent_url": filtered.parent_url,
            "filtered_out": filtered.filtered_out,
            "filter_reason": filtered.filter_reason,
            "metadata": filtered.metadata,
        }


class JSONLTraceWriter(TraceWriter):
    """Write trace as JSONL (one record per line for streaming/processing)."""
    
    def write(self, trace: SearchCrawlTrace, path: Path) -> None:
        """Write trace as JSONL.
        
        Each line is a JSON object representing either a PageRecord
        or FilteredCandidate, so downstream processors can stream
        and distinguish crawled vs excluded.
        
        Args:
            trace: SearchCrawlTrace to write
            path: Output JSONL file path
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            # Write header/metadata line
            header = {
                "type": "trace_header",
                "original_query": trace.original_query,
                "total_rounds": len(trace.rounds),
                "config_snapshot": trace.config_snapshot,
            }
            f.write(json.dumps(header, ensure_ascii=False) + "\n")
            
            # Write each round
            for round_data in trace.rounds:
                round_header = {
                    "type": "round_header",
                    "round_number": round_data.round_number,
                    "query": round_data.query,
                    "total_results": len(round_data.search_results.results),
                }
                f.write(json.dumps(round_header, ensure_ascii=False) + "\n")
                
                # Write pages (crawled)
                for page in round_data.pages:
                    page_dict = {
                        "type": "page",
                        "url": page.url,
                        "summary": page.summary,
                        "source_query": page.source_query,
                        "source_position": page.source_position,
                        "parent_url": page.parent_url,
                        "links_found": page.links_found,
                        "extracted_data": page.extracted_data,
                        "crawl_status": "success" if page.crawl_result and page.crawl_result.success else "failed",
                        "status_code": page.crawl_result.status_code if page.crawl_result else None,
                        "filtered_out": False,
                    }
                    f.write(json.dumps(page_dict, ensure_ascii=False) + "\n")
                
                # Write filtered candidates
                for filtered in round_data.filtered_out:
                    filtered_dict = {
                        "type": "filtered",
                        "url": filtered.url,
                        "title": filtered.title,
                        "snippet": filtered.snippet,
                        "source_query": filtered.source_query,
                        "source_position": filtered.source_position,
                        "parent_url": filtered.parent_url,
                        "filtered_out": True,
                        "filter_reason": filtered.filter_reason,
                        "metadata": filtered.metadata,
                    }
                    f.write(json.dumps(filtered_dict, ensure_ascii=False) + "\n")


class MarkdownTraceWriter(TraceWriter):
    """Write trace as markdown (human-readable summary)."""
    
    def write(self, trace: SearchCrawlTrace, path: Path) -> None:
        """Write trace as markdown.
        
        Args:
            trace: SearchCrawlTrace to write
            path: Output markdown file path
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# Search-Crawl Trace\n\n")
            f.write(f"**Original Query:** {trace.original_query}\n\n")
            f.write(f"**Total Rounds:** {len(trace.rounds)}\n\n")
            
            for round_data in trace.rounds:
                f.write(f"## Round {round_data.round_number}\n\n")
                f.write(f"**Query:** {round_data.query}\n\n")
                f.write(f"**Search Results:** {len(round_data.search_results.results)} found\n\n")
                
                if round_data.pages:
                    f.write(f"### Crawled Pages ({len(round_data.pages)})\n\n")
                    for i, page in enumerate(round_data.pages, 1):
                        f.write(f"{i}. **{page.url}**\n")
                        if page.summary:
                            f.write(f"   Summary: {page.summary[:200]}...\n")
                        if page.source_query:
                            f.write(f"   From query: {page.source_query} (position {page.source_position})\n")
                        if page.parent_url:
                            f.write(f"   Linked from: {page.parent_url}\n")
                        f.write("\n")
                
                if round_data.filtered_out:
                    f.write(f"### Filtered Out ({len(round_data.filtered_out)})\n\n")
                    for i, filtered in enumerate(round_data.filtered_out, 1):
                        f.write(f"{i}. **{filtered.url}**\n")
                        if filtered.title:
                            f.write(f"   Title: {filtered.title}\n")
                        f.write(f"   Reason: {filtered.filter_reason}\n")
                        f.write("\n")
                
                f.write("\n---\n\n")


def write_trace(
    trace: SearchCrawlTrace,
    path: Path,
    format: Literal["json", "markdown", "jsonl"] = "json",
) -> None:
    """Write trace to file in specified format.
    
    Args:
        trace: SearchCrawlTrace to write
        path: Output file path
        format: Output format (json, markdown, or jsonl)
    """
    writers = {
        "json": JSONTraceWriter(),
        "markdown": MarkdownTraceWriter(),
        "jsonl": JSONLTraceWriter(),
    }
    
    if format not in writers:
        raise ValueError(f"Unknown format: {format}. Must be one of: json, markdown, jsonl")
    
    writers[format].write(trace, path)

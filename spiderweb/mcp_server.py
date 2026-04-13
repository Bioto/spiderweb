"""MCP server for Spiderweb crawling and search capabilities.

Exposes crawl and search-crawl tools via Model Context Protocol (MCP).
This module is only loaded when the MCP server is invoked.
"""

import asyncio
from pathlib import Path
from typing import Any

from spiderweb.api import Spiderweb
from spiderweb.config import settings
from spiderweb.models.config import SearchDepthConfig


def _get_spiderweb_instance(vector_store_url: str | None = None) -> Spiderweb:
    """Create a Spiderweb instance with optional LLM and vector store.

    Args:
        vector_store_url: Optional vector store URL (e.g. qdrant://localhost:6333/my_collection)

    Returns:
        Spiderweb instance
    """
    from superglue import GlueLLM

    llm = None
    try:
        llm = GlueLLM(embedding_model=settings.embedding_model)
    except Exception:
        pass

    return Spiderweb(llm_client=llm, vector_store_url=vector_store_url)


def _serialize_crawl_result(result: Any, include_preview: bool = True) -> dict[str, Any]:
    """Convert CrawlResult to JSON-serializable dict.

    Args:
        result: CrawlResult object
        include_preview: If True, include markdown preview (truncated)

    Returns:
        Dict with url, success, title, markdown_preview, links_count, error
    """
    if not hasattr(result, "url"):
        return {
            "url": str(result) if result else "unknown",
            "success": False,
            "title": None,
            "markdown_preview": None,
            "links_count": 0,
            "error": "Invalid result object",
        }

    preview = None
    if include_preview and hasattr(result, "markdown") and result.markdown:
        preview = result.markdown[:2000] + "..." if len(result.markdown) > 2000 else result.markdown

    title = None
    if hasattr(result, "metadata") and isinstance(result.metadata, dict):
        title = result.metadata.get("title")
    elif hasattr(result, "title"):
        title = result.title

    links_count = 0
    if hasattr(result, "links") and result.links:
        links_count = len(result.links)

    error = None
    if hasattr(result, "success") and not result.success:
        error = getattr(result, "error", "Crawl failed") or "Crawl failed"

    return {
        "url": result.url,
        "success": getattr(result, "success", True),
        "title": title,
        "markdown_preview": preview,
        "links_count": links_count,
        "error": error,
    }


def create_mcp_server() -> Any:
    """Create and configure the FastMCP server with Spiderweb tools.

    Returns:
        FastMCP server instance with tools registered
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as e:
        raise ImportError(
            "MCP package not installed. Install with: pip install spiderweb[mcp]"
        ) from e

    mcp = FastMCP("Spiderweb", json_response=True)

    @mcp.tool()
    def crawl_url(
        url: str,
        save_to: str | None = None,
        save_format: str = "all",
        vector_store_url: str | None = None,
    ) -> dict[str, Any]:
        """Crawl a single URL and return the result.

        Args:
            url: URL to crawl
            save_to: Optional directory to save crawled content
            save_format: Format for saved files (markdown, html, json, all)
            vector_store_url: Optional vector store URL for ingestion

        Returns:
            Dict with url, success, title, markdown_preview, links_count, error
        """
        web = _get_spiderweb_instance(vector_store_url=vector_store_url)
        save_path = Path(save_to) if save_to else None

        try:
            result = asyncio.run(
                web.crawl_one(
                    url=url,
                    save_to=save_path,
                    save_format=save_format,
                )
            )
            return _serialize_crawl_result(result)
        except Exception as e:
            return {
                "url": url,
                "success": False,
                "title": None,
                "markdown_preview": None,
                "links_count": 0,
                "error": str(e),
            }

    @mcp.tool()
    def crawl_urls(
        urls: list[str],
        save_to: str | None = None,
        save_format: str = "all",
        vector_store_url: str | None = None,
    ) -> dict[str, Any]:
        """Crawl multiple URLs and return results.

        Args:
            urls: List of URLs to crawl
            save_to: Optional directory to save crawled content
            save_format: Format for saved files (markdown, html, json, all)
            vector_store_url: Optional vector store URL for ingestion

        Returns:
            Dict with results list (each item same as crawl_url return)
        """
        web = _get_spiderweb_instance(vector_store_url=vector_store_url)
        save_path = Path(save_to) if save_to else None

        try:
            results = asyncio.run(
                web.crawl_many(
                    urls=urls,
                    save_to=save_path,
                    save_format=save_format,
                )
            )
            return {
                "results": [_serialize_crawl_result(r) for r in results],
            }
        except Exception as e:
            return {
                "results": [],
                "error": str(e),
            }

    @mcp.tool()
    def search_and_crawl(
        query: str,
        max_rounds: int = 1,
        crawl_per_round: int = 3,
        save_to: str | None = None,
        save_trace_to: str | None = None,
        vector_store_url: str | None = None,
    ) -> dict[str, Any]:
        """Search the web, crawl results, and return summary.

        Args:
            query: Search query
            max_rounds: Maximum number of search rounds
            crawl_per_round: Number of search results to crawl per round
            save_to: Optional directory to save crawled content
            save_trace_to: Optional path to save search-crawl trace
            vector_store_url: Optional vector store URL for ingestion

        Returns:
            Dict with query, rounds_count, urls_crawled, urls_filtered, summaries
        """
        web = _get_spiderweb_instance(vector_store_url=vector_store_url)
        save_path = Path(save_to) if save_to else None
        trace_path = Path(save_trace_to) if save_trace_to else None

        try:
            depth_config = SearchDepthConfig(
                max_search_rounds=max_rounds,
                crawl_results_per_round=crawl_per_round,
            )

            trace = asyncio.run(
                web.search_crawl_extract(
                    query=query,
                    depth_config=depth_config,
                    save_to=save_path,
                    save_trace_to=trace_path,
                )
            )

            summaries = []
            for round_data in trace.rounds:
                for page in round_data.pages:
                    summaries.append(
                        {
                            "url": page.url,
                            "summary": page.summary[:500] + "..." if page.summary and len(page.summary) > 500 else page.summary,
                            "source_query": page.source_query,
                        }
                    )

            return {
                "query": query,
                "rounds_count": len(trace.rounds),
                "urls_crawled": trace.get_all_urls(),
                "urls_filtered": trace.get_all_filtered_urls(),
                "summaries": summaries,
            }
        except Exception as e:
            return {
                "query": query,
                "rounds_count": 0,
                "urls_crawled": [],
                "urls_filtered": [],
                "summaries": [],
                "error": str(e),
            }

    return mcp

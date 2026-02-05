"""REST API server for Spiderweb crawling and search capabilities.

Exposes crawl and search-crawl endpoints via FastAPI.
This module is only loaded when the REST API server is invoked.
"""

import asyncio
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from spiderweb.api import Spiderweb
from spiderweb.config import settings
from spiderweb.models.config import ChunkerConfig, QueryExpansionConfig, SearchDepthConfig
from spiderweb.models.result import BatchIngestionResult, IngestionResult, QueryResult


def _get_spiderweb_instance(vector_store_url: str | None = None) -> Spiderweb:
    """Create a Spiderweb instance with optional LLM and vector store.

    Args:
        vector_store_url: Optional vector store URL (e.g. qdrant://localhost:6333/my_collection)

    Returns:
        Spiderweb instance
    """
    from gluellm import GlueLLM

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


# Request/Response models
class CrawlRequest(BaseModel):
    """Request model for crawling a single URL."""

    url: str = Field(..., description="URL to crawl")
    save_to: str | None = Field(None, description="Optional directory to save crawled content")
    save_format: str = Field("all", description="Format for saved files (markdown, html, json, all)")
    vector_store_url: str | None = Field(None, description="Optional vector store URL for ingestion")


class CrawlResponse(BaseModel):
    """Response model for crawl operations."""

    url: str
    success: bool
    title: str | None = None
    markdown_preview: str | None = None
    links_count: int = 0
    error: str | None = None


class CrawlBatchRequest(BaseModel):
    """Request model for crawling multiple URLs."""

    urls: list[str] = Field(..., description="List of URLs to crawl")
    save_to: str | None = Field(None, description="Optional directory to save crawled content")
    save_format: str = Field("all", description="Format for saved files (markdown, html, json, all)")
    vector_store_url: str | None = Field(None, description="Optional vector store URL for ingestion")


class CrawlBatchResponse(BaseModel):
    """Response model for batch crawl operations."""

    results: list[CrawlResponse]
    error: str | None = None


class SearchRequest(BaseModel):
    """Request model for search and crawl."""

    query: str = Field(..., description="Search query")
    max_rounds: int = Field(1, description="Maximum number of search rounds")
    crawl_per_round: int = Field(3, description="Number of search results to crawl per round")
    save_to: str | None = Field(None, description="Optional directory to save crawled content")
    save_trace_to: str | None = Field(None, description="Optional path to save search-crawl trace")
    vector_store_url: str | None = Field(None, description="Optional vector store URL for ingestion")


class SearchResponse(BaseModel):
    """Response model for search and crawl."""

    query: str
    rounds_count: int
    urls_crawled: list[str]
    urls_filtered: list[str]
    summaries: list[dict[str, Any]]
    error: str | None = None


class IngestRequest(BaseModel):
    """Request model for ingesting a document."""

    path: str = Field(..., description="Path to file or directory to ingest")
    chunker: str = Field("hierarchical", description="Chunking strategy")
    chunk_size: int = Field(1000, description="Maximum chunk size")
    chunk_overlap: int = Field(200, description="Chunk overlap")
    recursive: bool = Field(True, description="Process subdirectories recursively")
    vector_store_url: str | None = Field(None, description="Optional vector store URL")
    force: bool = Field(False, description="Force re-processing (bypass cache)")


class IngestResponse(BaseModel):
    """Response model for ingestion."""

    success: bool
    total_documents: int | None = None
    successful_documents: int | None = None
    failed_documents: int | None = None
    total_chunks: int | None = None
    processing_time_seconds: float | None = None
    error: str | None = None


class QueryRequest(BaseModel):
    """Request model for querying the vector store."""

    query: str = Field(..., description="Search query")
    top_k: int = Field(5, description="Number of results to return")
    filter: dict[str, Any] | None = Field(None, description="Metadata filter")
    expand: bool = Field(False, description="Enable query expansion")
    expand_strategy: str = Field("multi_query", description="Query expansion strategy")
    expand_num: int = Field(3, description="Number of query expansions")
    vector_store_url: str | None = Field(None, description="Optional vector store URL")


class QueryResponse(BaseModel):
    """Response model for query results."""

    query: str
    chunks: list[dict[str, Any]]
    expanded_queries: list[str] | None = None
    error: str | None = None


def create_app() -> Any:
    """Create and configure the FastAPI app with Spiderweb endpoints.

    Returns:
        FastAPI app instance with routes registered
    """
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as e:
        raise ImportError(
            "FastAPI package not installed. Install with: pip install spiderweb[api]"
        ) from e

    app = FastAPI(
        title="Spiderweb API",
        description="REST API for web crawling and search capabilities",
        version="0.1.0",
    )

    @app.post("/crawl", response_model=CrawlResponse)
    async def crawl_url(request: CrawlRequest) -> CrawlResponse:
        """Crawl a single URL and return the result."""
        web = _get_spiderweb_instance(vector_store_url=request.vector_store_url)
        save_path = Path(request.save_to) if request.save_to else None

        try:
            result = await web.crawl_one(
                url=request.url,
                save_to=save_path,
                save_format=request.save_format,
            )
            serialized = _serialize_crawl_result(result)
            return CrawlResponse(**serialized)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/crawl/batch", response_model=CrawlBatchResponse)
    async def crawl_urls(request: CrawlBatchRequest) -> CrawlBatchResponse:
        """Crawl multiple URLs and return results."""
        web = _get_spiderweb_instance(vector_store_url=request.vector_store_url)
        save_path = Path(request.save_to) if request.save_to else None

        try:
            results = await web.crawl_many(
                urls=request.urls,
                save_to=save_path,
                save_format=request.save_format,
            )
            serialized_results = [_serialize_crawl_result(r) for r in results]
            return CrawlBatchResponse(
                results=[CrawlResponse(**r) for r in serialized_results]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/search", response_model=SearchResponse)
    async def search_and_crawl(request: SearchRequest) -> SearchResponse:
        """Search the web, crawl results, and return summary."""
        web = _get_spiderweb_instance(vector_store_url=request.vector_store_url)
        save_path = Path(request.save_to) if request.save_to else None
        trace_path = Path(request.save_trace_to) if request.save_trace_to else None

        try:
            depth_config = SearchDepthConfig(
                max_search_rounds=request.max_rounds,
                crawl_results_per_round=request.crawl_per_round,
            )

            trace = await web.search_crawl_extract(
                query=request.query,
                depth_config=depth_config,
                save_to=save_path,
                save_trace_to=trace_path,
            )

            summaries = []
            for round_data in trace.rounds:
                for page in round_data.pages:
                    summaries.append(
                        {
                            "url": page.url,
                            "summary": (
                                page.summary[:500] + "..."
                                if page.summary and len(page.summary) > 500
                                else page.summary
                            ),
                            "source_query": page.source_query,
                        }
                    )

            return SearchResponse(
                query=request.query,
                rounds_count=len(trace.rounds),
                urls_crawled=trace.get_all_urls(),
                urls_filtered=trace.get_all_filtered_urls(),
                summaries=summaries,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/ingest", response_model=IngestResponse)
    async def ingest_document(request: IngestRequest) -> IngestResponse:
        """Ingest a document or directory into the vector store."""
        web = _get_spiderweb_instance(vector_store_url=request.vector_store_url)
        path_obj = Path(request.path)

        if not path_obj.exists():
            raise HTTPException(status_code=404, detail=f"Path not found: {request.path}")

        try:
            # Set batch config for force flag
            if request.force:
                web.batch_processor.config.force = True

            # Create chunker config
            chunker_config = ChunkerConfig(
                strategy=request.chunker,
                max_chunk_size=request.chunk_size,
                chunk_overlap=request.chunk_overlap,
            )
            web.chunker_config = chunker_config

            if path_obj.is_file():
                result = await web.ingest(path_obj)
                return IngestResponse(
                    success=result.success,
                    total_documents=1,
                    successful_documents=1 if result.success else 0,
                    failed_documents=0 if result.success else 1,
                    total_chunks=result.chunks_created if result.success else 0,
                    processing_time_seconds=result.processing_time_seconds,
                    error=result.errors[0] if result.errors else None,
                )
            else:
                result = await web.ingest_directory(
                    path_obj,
                    recursive=request.recursive,
                    show_progress=False,
                )
                return IngestResponse(
                    success=result.successful_documents > 0,
                    total_documents=result.total_documents,
                    successful_documents=result.successful_documents,
                    failed_documents=result.failed_documents,
                    total_chunks=result.total_chunks,
                    processing_time_seconds=result.processing_time_seconds,
                )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/query", response_model=QueryResponse)
    async def query_index(request: QueryRequest) -> QueryResponse:
        """Query the vector store."""
        web = _get_spiderweb_instance(vector_store_url=request.vector_store_url)

        try:
            query_expansion = None
            if request.expand:
                query_expansion = QueryExpansionConfig(
                    enabled=True,
                    strategy=request.expand_strategy,
                    num_expansions=request.expand_num,
                )

            result = await web.query(
                request.query,
                top_k=request.top_k,
                filter_dict=request.filter,
                query_expansion=query_expansion,
            )

            # Serialize chunks
            chunks = []
            for chunk in result.chunks:
                chunks.append(
                    {
                        "content": chunk.get("content", ""),
                        "score": chunk.get("score"),
                        "metadata": chunk.get("metadata", {}),
                    }
                )

            return QueryResponse(
                query=request.query,
                chunks=chunks,
                expanded_queries=getattr(result, "expanded_queries", None),
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics() -> dict[str, Any]:
        """Get metrics endpoint (Prometheus-style).

        Returns metrics collected from Spiderweb operations including
        counters, histograms, and gauges.
        """
        from spiderweb.observability.metrics import get_metrics_collector

        collector = get_metrics_collector()
        return collector.get_metrics()

    return app

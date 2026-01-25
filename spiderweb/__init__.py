"""Spiderweb - Scalable Document Processing and RAG Pipeline.

Spiderweb is a document processing library built for RAG (Retrieval-Augmented Generation)
applications that provides:

- Multi-strategy document extraction with cross-validation
- Intelligent chunking (semantic, hierarchical, sliding window, etc.)
- Quality validation and semantic deduplication
- Pluggable vector storage (Qdrant, and more)
- Batch processing for hundreds of thousands of files
- Built on gluellm for LLM and embedding capabilities

Quick Start:
    >>> import asyncio
    >>> from spiderweb import ingest, query
    >>> from gluellm import GlueLLM
    >>>
    >>> async def main():
    ...     llm = GlueLLM()
    ...
    ...     # Ingest a document
    ...     doc = await ingest("document.pdf", llm_client=llm)
    ...     print(f"Created {len(doc.chunks)} chunks")
    ...
    ...     # Query the vector store
    ...     results = await query("What is the main topic?", llm_client=llm, top_k=5)
    ...     for chunk in results.chunks:
    ...         print(chunk["content"])
    ...
    >>> asyncio.run(main())

Main Components:
    - Spiderweb: Main client class for document processing
    - ingest: Quick document ingestion
    - query: Query the vector store
    - process_directory: Batch process a directory of documents

Configuration:
    - SpiderwebSettings: Global settings manager
    - ChunkerConfig: Chunking strategy configuration
    - ValidatorConfig: Validation pipeline configuration
"""

from spiderweb.api import Spiderweb, ingest, process_directory, query
from spiderweb.config import SpiderwebSettings, get_settings, settings
from spiderweb.models.config import ChunkerConfig, ValidatorConfig, VectorStoreConfig

__all__ = [
    # Main API
    "Spiderweb",
    "ingest",
    "query",
    "process_directory",
    # Configuration
    "SpiderwebSettings",
    "settings",
    "get_settings",
    "ChunkerConfig",
    "ValidatorConfig",
    "VectorStoreConfig",
]

__version__ = "0.1.0"

"""Core document and chunk data models.

These models represent the fundamental data structures used throughout Spiderweb
for document processing and chunk management.
"""

import hashlib
import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChunkType(str, Enum):
    """Enumeration of supported chunking strategies."""

    SEMANTIC = "semantic"
    HIERARCHICAL = "hierarchical"
    SLIDING_WINDOW = "sliding_window"
    SENTENCE = "sentence"
    RECURSIVE = "recursive"
    CUSTOM = "custom"


class DocumentMetadata(BaseModel):
    """Metadata associated with a document.

    Contains information about the document's source, extraction method,
    and any additional context.
    """

    source: str = Field(
        description="File path, URL, or identifier of the document source",
    )
    file_type: str = Field(
        description="Document file type (e.g., 'pdf', 'docx', 'md')",
    )
    extraction_method: str = Field(
        description="Method used to extract content (e.g., 'markitdown', 'pdfminer')",
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when the document was processed",
    )
    file_size_bytes: int | None = Field(
        default=None,
        description="Original file size in bytes",
    )
    page_count: int | None = Field(
        default=None,
        description="Number of pages (for documents with pages)",
    )
    language: str | None = Field(
        default=None,
        description="Detected or specified document language",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional custom metadata",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "source": "/path/to/document.pdf",
                "file_type": "pdf",
                "extraction_method": "markitdown",
                "created_at": "2024-01-01T12:00:00",
                "file_size_bytes": 1024000,
                "page_count": 10,
                "language": "en",
                "extra": {"author": "John Doe", "department": "Engineering"},
            }
        }
    )


class ChunkMetadata(BaseModel):
    """Metadata associated with a chunk.

    Contains information about the chunk's position within the document,
    its chunking strategy, and any additional context.
    """

    document_id: str = Field(
        description="ID of the parent document",
    )
    chunk_index: int = Field(
        description="Index of this chunk within the document (0-based)",
    )
    chunk_type: ChunkType = Field(
        description="Chunking strategy used to create this chunk",
    )
    start_char: int | None = Field(
        default=None,
        description="Starting character position in the original document",
    )
    end_char: int | None = Field(
        default=None,
        description="Ending character position in the original document",
    )
    page_numbers: list[int] = Field(
        default_factory=list,
        description="Page numbers this chunk spans (for paginated documents)",
    )
    section_title: str | None = Field(
        default=None,
        description="Section or heading title (for hierarchical chunks)",
    )
    section_level: int | None = Field(
        default=None,
        description="Hierarchical level (0=root, 1=chapter, 2=section, etc.)",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional custom metadata",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "document_id": "doc_123",
                "chunk_index": 0,
                "chunk_type": "semantic",
                "start_char": 0,
                "end_char": 1000,
                "page_numbers": [1, 2],
                "section_title": "Introduction",
                "section_level": 1,
                "extra": {},
            }
        }
    )


class Chunk(BaseModel):
    """A chunk of text from a document.

    Represents a piece of a document that has been chunked according to
    some strategy. May contain embeddings and validation scores.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this chunk",
    )
    content: str = Field(
        description="The text content of the chunk",
        min_length=1,
    )
    embedding: list[float] | None = Field(
        default=None,
        description="Vector embedding of the chunk content",
    )
    metadata: ChunkMetadata = Field(
        description="Metadata about this chunk",
    )
    validation_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Validation scores (e.g., quality: 0.8, coherence: 0.9)",
    )
    parent_id: str | None = Field(
        default=None,
        description="ID of the parent chunk (for hierarchical chunking)",
    )
    children_ids: list[str] = Field(
        default_factory=list,
        description="IDs of child chunks (for hierarchical chunking)",
    )

    @field_validator("content")
    @classmethod
    def validate_content_not_empty(cls, v: str) -> str:
        """Ensure chunk content is not empty or just whitespace."""
        if not v or not v.strip():
            raise ValueError("Chunk content cannot be empty")
        return v

    @field_validator("embedding")
    @classmethod
    def validate_embedding_dimension(cls, v: list[float] | None) -> list[float] | None:
        """Ensure embedding has reasonable dimensions if present."""
        if v is not None:
            if len(v) == 0:
                raise ValueError("Embedding cannot be empty list")
            if len(v) > 10000:
                raise ValueError(f"Embedding dimension {len(v)} is unreasonably large")
        return v

    def content_hash(self) -> str:
        """Generate a hash of the chunk content for deduplication.

        Returns:
            SHA-256 hash of the chunk content
        """
        return hashlib.sha256(self.content.encode()).hexdigest()

    def character_count(self) -> int:
        """Count the number of characters in the chunk.

        Returns:
            Character count
        """
        return len(self.content)

    def word_count(self) -> int:
        """Count the number of words in the chunk.

        Returns:
            Word count (simple whitespace-based split)
        """
        return len(self.content.split())

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "chunk_abc123",
                "content": "This is a sample chunk of text from a document.",
                "embedding": [0.1, 0.2, 0.3],  # Simplified for example
                "metadata": {
                    "document_id": "doc_123",
                    "chunk_index": 0,
                    "chunk_type": "semantic",
                },
                "validation_scores": {"quality": 0.85, "coherence": 0.92},
                "parent_id": None,
                "children_ids": [],
            }
        }
    )


class Document(BaseModel):
    """A document with its content and chunks.

    Represents a complete document that has been processed through extraction
    and chunking. Contains the original content, normalized markdown, and
    generated chunks.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this document",
    )
    raw_content: str = Field(
        description="Original extracted text content",
    )
    markdown_content: str = Field(
        description="Normalized markdown version of the content",
    )
    chunks: list[Chunk] = Field(
        default_factory=list,
        description="List of chunks generated from this document",
    )
    metadata: DocumentMetadata = Field(
        description="Document metadata",
    )
    extraction_results: dict[str, str] = Field(
        default_factory=dict,
        description="Content from different extraction methods (method_name -> content)",
    )

    @field_validator("chunks")
    @classmethod
    def validate_chunk_indices(cls, v: list[Chunk]) -> list[Chunk]:
        """Ensure chunk indices are sequential if present."""
        if v:
            indices = [chunk.metadata.chunk_index for chunk in v]
            if indices and max(indices) >= len(v):
                raise ValueError("Chunk indices are not properly sequential")
        return v

    def total_chunks(self) -> int:
        """Get the total number of chunks.

        Returns:
            Number of chunks
        """
        return len(self.chunks)

    def character_count(self) -> int:
        """Count total characters in the document.

        Returns:
            Character count
        """
        return len(self.raw_content)

    def word_count(self) -> int:
        """Count total words in the document.

        Returns:
            Word count (simple whitespace-based split)
        """
        return len(self.raw_content.split())

    def chunks_with_embeddings(self) -> list[Chunk]:
        """Get only chunks that have embeddings.

        Returns:
            List of chunks with embeddings
        """
        return [chunk for chunk in self.chunks if chunk.embedding is not None]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "doc_123",
                "raw_content": "This is the raw extracted content...",
                "markdown_content": "# Document Title\n\nThis is the markdown version...",
                "chunks": [],
                "metadata": {
                    "source": "/path/to/document.pdf",
                    "file_type": "pdf",
                    "extraction_method": "markitdown",
                },
                "extraction_results": {
                    "markitdown": "Content via markitdown...",
                    "pdfminer": "Content via pdfminer...",
                },
            }
        }
    )

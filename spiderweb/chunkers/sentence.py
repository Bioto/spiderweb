"""Sentence-based chunker.

Splits documents into chunks at sentence boundaries.
"""

from spiderweb.chunkers.base import split_into_sentences
from spiderweb.models.config import ChunkerConfig
from spiderweb.models.document import Chunk, ChunkMetadata, ChunkType, Document
from spiderweb.observability.logging_config import get_logger

logger = get_logger(__name__)


class SentenceChunker:
    """Chunker that groups sentences into chunks.

    Creates chunks by grouping consecutive sentences until a size limit
    is reached, ensuring no sentence is split across chunks.

    Example:
        >>> chunker = SentenceChunker(max_chunk_size=1000)
        >>> chunks = chunker.chunk(document)
    """

    def __init__(self, max_chunk_size: int = 1000, min_chunk_size: int = 100):
        """Initialize sentence chunker.

        Args:
            max_chunk_size: Maximum size of each chunk in characters
            min_chunk_size: Minimum size of each chunk in characters
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

        logger.debug(f"Initialized SentenceChunker: max_size={max_chunk_size}, min_size={min_chunk_size}")

    @classmethod
    def from_config(cls, config: ChunkerConfig) -> "SentenceChunker":
        """Create chunker from configuration.

        Args:
            config: Chunker configuration

        Returns:
            Configured chunker instance
        """
        return cls(
            max_chunk_size=config.max_chunk_size,
            min_chunk_size=config.min_chunk_size,
        )

    def chunk(self, document: Document) -> list[Chunk]:
        """Split document into sentence-based chunks.

        Args:
            document: Document to chunk

        Returns:
            List of chunks
        """
        text = document.markdown_content

        if not text.strip():
            logger.warning(f"Document {document.id} has no content to chunk")
            return []

        sentences = split_into_sentences(text)

        if not sentences:
            logger.warning(f"No sentences found in document {document.id}")
            return []

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # If adding this sentence would exceed max size and we have content, finalize chunk
            if current_length + sentence_length > self.max_chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        # Add remaining chunk if it meets minimum size
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size or not chunks:
                chunks.append(chunk_text)
            elif chunks:
                # If too small, merge with previous chunk
                chunks[-1] = chunks[-1] + " " + chunk_text

        logger.info(f"Split document {document.id} into {len(chunks)} sentence-based chunks")

        # Create Chunk objects
        result_chunks = []
        for idx, chunk_text in enumerate(chunks):
            metadata = ChunkMetadata(
                document_id=document.id,
                chunk_index=idx,
                chunk_type=ChunkType.SENTENCE,
            )

            chunk = Chunk(
                content=chunk_text,
                metadata=metadata,
            )

            result_chunks.append(chunk)

        return result_chunks

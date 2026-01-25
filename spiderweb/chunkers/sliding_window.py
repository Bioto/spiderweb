"""Sliding window chunker.

Implements a simple sliding window strategy with configurable size and overlap.
"""

from spiderweb.chunkers.base import split_into_sentences
from spiderweb.models.config import ChunkerConfig
from spiderweb.models.document import Chunk, ChunkMetadata, ChunkType, Document
from spiderweb.observability.logging_config import get_logger

logger = get_logger(__name__)


class SlidingWindowChunker:
    """Chunker that uses a sliding window approach.

    Splits documents into fixed-size chunks with configurable overlap.
    Optionally respects sentence boundaries to avoid cutting mid-sentence.

    Example:
        >>> chunker = SlidingWindowChunker(max_chunk_size=1000, chunk_overlap=200)
        >>> chunks = chunker.chunk(document)
        >>> print(f"Created {len(chunks)} chunks")
    """

    def __init__(
        self,
        max_chunk_size: int = 1000,
        chunk_overlap: int = 200,
        respect_boundaries: bool = True,
    ):
        """Initialize sliding window chunker.

        Args:
            max_chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Overlap between consecutive chunks in characters
            respect_boundaries: Try to split at sentence boundaries
        """
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.respect_boundaries = respect_boundaries

        logger.debug(
            f"Initialized SlidingWindowChunker: max_size={max_chunk_size}, "
            f"overlap={chunk_overlap}, respect_boundaries={respect_boundaries}"
        )

    @classmethod
    def from_config(cls, config: ChunkerConfig) -> "SlidingWindowChunker":
        """Create chunker from configuration.

        Args:
            config: Chunker configuration

        Returns:
            Configured chunker instance
        """
        return cls(
            max_chunk_size=config.max_chunk_size,
            chunk_overlap=config.chunk_overlap,
            respect_boundaries=config.respect_boundaries,
        )

    def _split_with_boundaries(self, text: str) -> list[str]:
        """Split text into chunks respecting sentence boundaries.

        Args:
            text: Text to split

        Returns:
            List of chunk strings
        """
        sentences = split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # If adding this sentence would exceed max size, finalize current chunk
            if current_length + sentence_length > self.max_chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))

                # Calculate overlap
                overlap_text = []
                overlap_length = 0

                # Add sentences from the end until we reach overlap size
                for s in reversed(current_chunk):
                    s_len = len(s)
                    if overlap_length + s_len <= self.chunk_overlap:
                        overlap_text.insert(0, s)
                        overlap_length += s_len
                    else:
                        break

                current_chunk = overlap_text
                current_length = overlap_length

            current_chunk.append(sentence)
            current_length += sentence_length

        # Add remaining chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _split_simple(self, text: str) -> list[str]:
        """Split text into chunks without respecting boundaries.

        Args:
            text: Text to split

        Returns:
            List of chunk strings
        """
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + self.max_chunk_size, text_length)
            chunks.append(text[start:end])
            start = end - self.chunk_overlap if end < text_length else text_length

        return chunks

    def chunk(self, document: Document) -> list[Chunk]:
        """Split document into chunks using sliding window.

        Args:
            document: Document to chunk

        Returns:
            List of chunks
        """
        text = document.markdown_content

        if not text.strip():
            logger.warning(f"Document {document.id} has no content to chunk")
            return []

        # Split text into chunks
        chunk_texts = self._split_with_boundaries(text) if self.respect_boundaries else self._split_simple(text)

        logger.info(
            f"Split document {document.id} ({len(text)} chars) into {len(chunk_texts)} chunks "
            f"(avg: {len(text) // len(chunk_texts) if chunk_texts else 0} chars/chunk)"
        )

        # Create Chunk objects
        chunks = []
        for idx, chunk_text in enumerate(chunk_texts):
            # Calculate character positions (approximate for boundary-respecting mode)
            start_char = 0 if idx == 0 else sum(len(c) for c in chunk_texts[:idx]) - idx * self.chunk_overlap

            end_char = start_char + len(chunk_text)

            metadata = ChunkMetadata(
                document_id=document.id,
                chunk_index=idx,
                chunk_type=ChunkType.SLIDING_WINDOW,
                start_char=start_char,
                end_char=end_char,
            )

            chunk = Chunk(
                content=chunk_text,
                metadata=metadata,
            )

            chunks.append(chunk)

        return chunks

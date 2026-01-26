"""Semantic chunker using embeddings.

Splits documents based on semantic similarity between text segments.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gluellm import GlueLLM

from spiderweb.chunkers.base import split_into_sentences
from spiderweb.models.config import ChunkerConfig
from spiderweb.models.document import Chunk, ChunkMetadata, ChunkType, Document
from spiderweb.observability.logging_config import get_logger
from spiderweb.utils.vector_math import cosine_similarity

logger = get_logger(__name__)


class SemanticChunker:
    """Chunker that splits based on semantic similarity.

    Uses embeddings to determine when the topic/context changes significantly,
    creating natural semantic boundaries for chunks.

    Requires gluellm for embedding generation.

    Example:
        >>> from gluellm import GlueLLM
        >>> llm_client = GlueLLM()
        >>> chunker = SemanticChunker(llm_client=llm_client, threshold=0.7)
        >>> chunks = await chunker.chunk_async(document)
    """

    def __init__(
        self,
        llm_client: "GlueLLM | None" = None,
        threshold: float = 0.7,
        max_chunk_size: int = 2000,
        min_chunk_size: int = 200,
    ):
        """Initialize semantic chunker.

        Args:
            llm_client: GlueLLM client for generating embeddings
            threshold: Similarity threshold for chunk boundaries (0-1)
            max_chunk_size: Maximum chunk size in characters
            min_chunk_size: Minimum chunk size in characters
        """
        self.llm_client = llm_client
        self.threshold = threshold
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

        logger.debug(
            f"Initialized SemanticChunker: threshold={threshold}, max_size={max_chunk_size}, min_size={min_chunk_size}"
        )

    @classmethod
    def from_config(cls, config: ChunkerConfig, llm_client: "GlueLLM | None" = None) -> "SemanticChunker":
        """Create chunker from configuration.

        Args:
            config: Chunker configuration
            llm_client: GlueLLM client for embeddings

        Returns:
            Configured chunker instance
        """
        return cls(
            llm_client=llm_client,
            threshold=config.semantic_threshold,
            max_chunk_size=config.max_chunk_size,
            min_chunk_size=config.min_chunk_size,
        )

    async def chunk_async(self, document: Document) -> list[Chunk]:
        """Split document into semantic chunks (async version).

        Args:
            document: Document to chunk

        Returns:
            List of chunks

        Raises:
            ValueError: If llm_client is not provided
        """
        if self.llm_client is None:
            raise ValueError("llm_client is required for semantic chunking")

        text = document.markdown_content

        if not text.strip():
            logger.warning(f"Document {document.id} has no content to chunk")
            return []

        # Split into sentences
        sentences = split_into_sentences(text)

        if len(sentences) <= 1:
            # Single sentence or less - return as one chunk
            metadata = ChunkMetadata(
                document_id=document.id,
                chunk_index=0,
                chunk_type=ChunkType.SEMANTIC,
            )
            return [Chunk(content=text, metadata=metadata)]

        # Generate embeddings for sentences
        logger.debug(f"Generating embeddings for {len(sentences)} sentences")
        embedding_result = await self.llm_client.embed(sentences)
        embeddings = embedding_result.embeddings

        # Find semantic boundaries based on similarity
        boundaries = [0]  # Start of first chunk

        current_chunk = [sentences[0]]
        current_length = len(sentences[0])

        for i in range(1, len(sentences)):
            # Calculate similarity with previous sentence
            similarity = cosine_similarity(embeddings[i - 1], embeddings[i])

            sentence_length = len(sentences[i])

            # Create boundary if:
            # 1. Similarity drops below threshold (semantic shift), OR
            # 2. Adding this sentence would exceed max size
            if similarity < self.threshold or (current_length + sentence_length > self.max_chunk_size):
                # Only create boundary if current chunk meets minimum size
                if current_length >= self.min_chunk_size:
                    boundaries.append(i)
                    current_chunk = [sentences[i]]
                    current_length = sentence_length
                else:
                    current_chunk.append(sentences[i])
                    current_length += sentence_length
            else:
                current_chunk.append(sentences[i])
                current_length += sentence_length

        boundaries.append(len(sentences))  # End boundary

        # Create chunks from boundaries
        chunks = []
        for idx, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
            chunk_text = " ".join(sentences[start:end])

            metadata = ChunkMetadata(
                document_id=document.id,
                chunk_index=idx,
                chunk_type=ChunkType.SEMANTIC,
            )

            chunk = Chunk(
                content=chunk_text,
                metadata=metadata,
            )

            chunks.append(chunk)

        logger.info(
            f"Split document {document.id} into {len(chunks)} semantic chunks "
            f"(avg similarity threshold: {self.threshold})"
        )

        return chunks

    def chunk(self, document: Document) -> list[Chunk]:
        """Synchronous chunk method (not recommended for semantic chunking).

        This method is provided for compatibility but will raise an error.
        Use chunk_async() instead.

        Args:
            document: Document to chunk

        Raises:
            RuntimeError: Always, as semantic chunking requires async
        """
        raise RuntimeError(
            "Semantic chunking requires async operation. Use chunk_async() instead. "
            "Example: chunks = await chunker.chunk_async(document)"
        )

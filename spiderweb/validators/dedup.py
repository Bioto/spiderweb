"""Deduplication validator for chunks.

Detects duplicate and near-duplicate chunks using embeddings.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gluellm import GlueLLM

from spiderweb.models.config import ValidatorConfig
from spiderweb.models.document import Chunk
from spiderweb.models.result import ValidationResult
from spiderweb.observability.logging_config import get_logger

logger = get_logger(__name__)


class DedupValidator:
    """Validator that detects duplicate chunks.

    Uses content hashing for exact duplicates and embedding similarity
    for near-duplicates.

    Example:
        >>> from gluellm import GlueLLM
        >>> llm = GlueLLM()
        >>> validator = DedupValidator(llm_client=llm, threshold=0.95)
        >>> # Track seen chunks
        >>> for chunk in chunks:
        ...     result = await validator.validate(chunk)
        ...     if not result.is_duplicate:
        ...         validator.add_seen(chunk)
    """

    def __init__(
        self,
        llm_client: "GlueLLM | None" = None,
        threshold: float = 0.95,
        use_content_hash: bool = True,
        use_embedding_similarity: bool = True,
    ):
        """Initialize deduplication validator.

        Args:
            llm_client: GlueLLM client for generating embeddings
            threshold: Similarity threshold for near-duplicates (0-1)
            use_content_hash: Check exact duplicates via hash
            use_embedding_similarity: Check near-duplicates via embeddings
        """
        self.llm_client = llm_client
        self.threshold = threshold
        self.use_content_hash = use_content_hash
        self.use_embedding_similarity = use_embedding_similarity

        # Tracking seen chunks
        self._seen_hashes: set[str] = set()
        self._seen_chunks: list[tuple[str, list[float]]] = []  # (chunk_id, embedding)

        logger.debug(
            f"Initialized DedupValidator: threshold={threshold}, "
            f"hash={use_content_hash}, embedding={use_embedding_similarity}"
        )

    @classmethod
    def from_config(cls, config: ValidatorConfig, llm_client: "GlueLLM | None" = None) -> "DedupValidator":
        """Create validator from configuration.

        Args:
            config: Validator configuration
            llm_client: GlueLLM client for embeddings

        Returns:
            Configured validator instance
        """
        return cls(
            llm_client=llm_client,
            threshold=config.deduplication_threshold,
            use_content_hash=True,
            use_embedding_similarity=config.enable_deduplication,
        )

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity (0-1)
        """
        import math

        dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=True))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def add_seen(self, chunk: Chunk) -> None:
        """Add a chunk to the seen set.

        Call this after validating a chunk that should be kept.

        Args:
            chunk: Chunk to remember
        """
        if self.use_content_hash:
            self._seen_hashes.add(chunk.content_hash())

        if self.use_embedding_similarity and chunk.embedding:
            self._seen_chunks.append((chunk.id, chunk.embedding))

    def clear_seen(self) -> None:
        """Clear all seen chunks.

        Useful when starting a new batch of documents.
        """
        self._seen_hashes.clear()
        self._seen_chunks.clear()
        logger.debug("Cleared deduplication cache")

    async def validate(self, chunk: Chunk) -> ValidationResult:
        """Check if chunk is a duplicate.

        Args:
            chunk: Chunk to validate

        Returns:
            Validation result indicating if chunk is duplicate
        """
        is_duplicate = False
        duplicate_of = None

        # Check exact duplicates via hash
        if self.use_content_hash:
            chunk_hash = chunk.content_hash()
            if chunk_hash in self._seen_hashes:
                is_duplicate = True
                logger.debug(f"Chunk {chunk.id} is exact duplicate (hash match)")

        # Check near-duplicates via embedding similarity
        if not is_duplicate and self.use_embedding_similarity:
            # Generate embedding if not present
            if chunk.embedding is None and self.llm_client:
                embedding_result = await self.llm_client.embed(chunk.content)
                chunk.embedding = embedding_result.embeddings[0]

            if chunk.embedding:
                # Compare with seen chunks
                for seen_id, seen_embedding in self._seen_chunks:
                    similarity = self._cosine_similarity(chunk.embedding, seen_embedding)

                    if similarity >= self.threshold:
                        is_duplicate = True
                        duplicate_of = seen_id
                        logger.debug(f"Chunk {chunk.id} is near-duplicate of {seen_id} (similarity={similarity:.3f})")
                        break

        passed = not is_duplicate

        issues = []
        if is_duplicate:
            if duplicate_of:
                issues.append(f"Near-duplicate of chunk {duplicate_of}")
            else:
                issues.append("Exact duplicate")

        return ValidationResult(
            chunk_id=chunk.id,
            passed=passed,
            quality_score=0.0 if is_duplicate else 1.0,
            is_duplicate=is_duplicate,
            duplicate_of=duplicate_of,
            issues=issues,
            recommendations=["Remove duplicate"] if is_duplicate else [],
        )

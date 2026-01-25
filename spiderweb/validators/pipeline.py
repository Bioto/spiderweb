"""Validation pipeline for orchestrating multiple validators.

Coordinates quality checking, deduplication, and optional LLM validation.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gluellm import GlueLLM

from spiderweb.models.config import ValidatorConfig
from spiderweb.models.document import Chunk
from spiderweb.models.result import ValidationResult
from spiderweb.observability.logging_config import get_logger
from spiderweb.validators.dedup import DedupValidator
from spiderweb.validators.quality import QualityValidator

logger = get_logger(__name__)


class ValidationPipeline:
    """Pipeline for validating chunks through multiple validators.

    Orchestrates quality checking, deduplication, and optional LLM validation.

    Example:
        >>> from gluellm import GlueLLM
        >>> llm = GlueLLM()
        >>> pipeline = ValidationPipeline(llm_client=llm)
        >>> results = await pipeline.validate_batch(chunks)
        >>> valid_chunks = [c for c, r in zip(chunks, results) if r.passed]
    """

    def __init__(
        self,
        llm_client: "GlueLLM | None" = None,
        config: ValidatorConfig | None = None,
        enable_quality: bool = True,
        enable_deduplication: bool = True,
    ):
        """Initialize validation pipeline.

        Args:
            llm_client: GlueLLM client for embeddings and LLM validation
            config: Validator configuration
            enable_quality: Enable quality validation
            enable_deduplication: Enable deduplication
        """
        self.llm_client = llm_client
        self.config = config or ValidatorConfig()
        self.enable_quality = enable_quality and self.config.enable_validation
        self.enable_deduplication = enable_deduplication and self.config.enable_deduplication

        # Initialize validators
        self.quality_validator = QualityValidator.from_config(self.config) if self.enable_quality else None

        self.dedup_validator = (
            DedupValidator.from_config(self.config, llm_client=llm_client) if self.enable_deduplication else None
        )

        logger.info(
            f"Initialized ValidationPipeline: quality={self.enable_quality}, "
            f"dedup={self.enable_deduplication}, llm_validation={self.config.enable_llm_validation}"
        )

    async def validate(self, chunk: Chunk) -> ValidationResult:
        """Validate a single chunk through the pipeline.

        Args:
            chunk: Chunk to validate

        Returns:
            Combined validation result
        """
        # Start with base result
        result = ValidationResult(
            chunk_id=chunk.id,
            passed=True,
            quality_score=1.0,
            is_duplicate=False,
        )

        # Quality validation
        if self.quality_validator:
            quality_result = await self.quality_validator.validate(chunk)
            result.quality_score = quality_result.quality_score
            result.information_density = quality_result.information_density
            result.issues.extend(quality_result.issues)
            result.recommendations.extend(quality_result.recommendations)

            if not quality_result.passed:
                result.passed = False

        # Deduplication
        if self.dedup_validator:
            dedup_result = await self.dedup_validator.validate(chunk)

            if dedup_result.is_duplicate:
                result.is_duplicate = True
                result.duplicate_of = dedup_result.duplicate_of
                result.passed = False
                result.issues.extend(dedup_result.issues)
            else:
                # Add to seen set if not duplicate
                self.dedup_validator.add_seen(chunk)

        # Store validation scores in chunk
        chunk.validation_scores = {
            "quality": result.quality_score,
            "information_density": result.information_density or 0.0,
        }

        return result

    async def validate_batch(self, chunks: list[Chunk]) -> list[ValidationResult]:
        """Validate a batch of chunks.

        Args:
            chunks: List of chunks to validate

        Returns:
            List of validation results
        """
        logger.info(f"Validating batch of {len(chunks)} chunks")

        results = []
        for chunk in chunks:
            result = await self.validate(chunk)
            results.append(result)

        passed_count = sum(1 for r in results if r.passed)
        failed_count = len(results) - passed_count
        duplicate_count = sum(1 for r in results if r.is_duplicate)

        logger.info(f"Validation complete: {passed_count} passed, {failed_count} failed, {duplicate_count} duplicates")

        return results

    def reset(self) -> None:
        """Reset the validation pipeline state.

        Clears deduplication cache and any other stateful components.
        """
        if self.dedup_validator:
            self.dedup_validator.clear_seen()

        logger.debug("Reset validation pipeline")

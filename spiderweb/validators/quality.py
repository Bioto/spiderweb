"""Quality validator for chunks.

Scores chunks based on length, information density, and completeness.
"""

from spiderweb.models.config import ValidatorConfig
from spiderweb.models.document import Chunk
from spiderweb.models.result import ValidationResult
from spiderweb.observability.logging_config import get_logger

logger = get_logger(__name__)


class QualityValidator:
    """Validator that scores chunk quality.

    Evaluates chunks based on:
    - Length (not too short, not too long)
    - Information density (ratio of meaningful content)
    - Completeness (starts/ends at reasonable boundaries)

    Example:
        >>> validator = QualityValidator(min_score=0.5)
        >>> result = await validator.validate(chunk)
        >>> print(f"Quality score: {result.quality_score}")
    """

    def __init__(
        self,
        min_score: float = 0.3,
        min_length: int = 50,
        max_length: int = 5000,
        min_information_density: float = 0.2,
    ):
        """Initialize quality validator.

        Args:
            min_score: Minimum acceptable quality score (0-1)
            min_length: Minimum chunk length in characters
            max_length: Maximum chunk length in characters
            min_information_density: Minimum information density (0-1)
        """
        self.min_score = min_score
        self.min_length = min_length
        self.max_length = max_length
        self.min_information_density = min_information_density

        logger.debug(f"Initialized QualityValidator: min_score={min_score}, min_length={min_length}")

    @classmethod
    def from_config(cls, config: ValidatorConfig) -> "QualityValidator":
        """Create validator from configuration.

        Args:
            config: Validator configuration

        Returns:
            Configured validator instance
        """
        return cls(
            min_score=config.min_quality_score,
            min_information_density=config.min_information_density,
        )

    def _calculate_length_score(self, chunk: Chunk) -> float:
        """Calculate score based on chunk length.

        Args:
            chunk: Chunk to score

        Returns:
            Length score (0-1), where 1.0 is optimal
        """
        length = len(chunk.content)

        if length < self.min_length:
            # Too short - penalize
            return length / self.min_length * 0.5
        if length > self.max_length:
            # Too long - penalize
            return max(0.0, 1.0 - (length - self.max_length) / self.max_length)
        # Good length
        # Optimal around 1000 characters
        optimal = 1000
        distance = abs(length - optimal) / optimal
        return max(0.5, 1.0 - distance * 0.5)

    def _calculate_information_density(self, chunk: Chunk) -> float:
        """Calculate information density score.

        Measures ratio of meaningful content (letters/digits) to total characters.

        Args:
            chunk: Chunk to score

        Returns:
            Information density score (0-1)
        """
        content = chunk.content

        # Count meaningful characters (letters, digits)
        meaningful = sum(1 for c in content if c.isalnum())
        total = len(content)

        if total == 0:
            return 0.0

        density = meaningful / total

        # Normalize: typical good text has ~0.7-0.8 density
        # Below 0.2 is likely mostly formatting/whitespace
        return min(1.0, density / 0.75)

    def _check_completeness(self, chunk: Chunk) -> tuple[bool, list[str]]:
        """Check if chunk appears complete.

        Args:
            chunk: Chunk to check

        Returns:
            Tuple of (is_complete, list of issues)
        """
        content = chunk.content.strip()
        issues = []

        # Check if starts with lowercase (might be mid-sentence)
        if content and content[0].islower():
            issues.append("Starts with lowercase letter (incomplete sentence)")

        # Check if ends without punctuation
        if content and content[-1] not in ".!?\"'":
            issues.append("Does not end with proper punctuation")

        # Check for very unbalanced quotes
        double_quotes = content.count('"')

        if double_quotes % 2 != 0:
            issues.append("Unbalanced double quotes")

        # Check for unbalanced parentheses
        open_parens = content.count("(")
        close_parens = content.count(")")

        if open_parens != close_parens:
            issues.append(f"Unbalanced parentheses ({open_parens} open, {close_parens} close)")

        is_complete = len(issues) == 0

        return is_complete, issues

    async def validate(self, chunk: Chunk) -> ValidationResult:
        """Validate chunk quality.

        Args:
            chunk: Chunk to validate

        Returns:
            Validation result with quality scores
        """
        # Calculate component scores
        length_score = self._calculate_length_score(chunk)
        density = self._calculate_information_density(chunk)
        is_complete, completeness_issues = self._check_completeness(chunk)

        # Calculate overall quality score (weighted average)
        quality_score = length_score * 0.3 + density * 0.4 + (1.0 if is_complete else 0.5) * 0.3

        # Determine if passed
        passed = quality_score >= self.min_score and density >= self.min_information_density

        issues = []
        recommendations = []

        if length_score < 0.7:
            if len(chunk.content) < self.min_length:
                issues.append(f"Chunk too short ({len(chunk.content)} < {self.min_length})")
                recommendations.append("Merge with adjacent chunks")
            else:
                issues.append(f"Chunk too long ({len(chunk.content)} > {self.max_length})")
                recommendations.append("Split into smaller chunks")

        if density < self.min_information_density:
            issues.append(f"Low information density ({density:.2f})")
            recommendations.append("Remove excessive whitespace or formatting")

        issues.extend(completeness_issues)

        if not passed:
            logger.debug(
                f"Chunk {chunk.id} failed quality validation (score={quality_score:.2f}, density={density:.2f})"
            )

        return ValidationResult(
            chunk_id=chunk.id,
            passed=passed,
            quality_score=quality_score,
            information_density=density,
            issues=issues,
            recommendations=recommendations,
        )

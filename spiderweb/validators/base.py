"""Base validator protocol.

Defines the interface for chunk validators.
"""

from typing import Protocol, runtime_checkable

from spiderweb.models.document import Chunk
from spiderweb.models.result import ValidationResult


@runtime_checkable
class Validator(Protocol):
    """Protocol for chunk validators.

    All validators must implement this interface.
    """

    async def validate(self, chunk: Chunk) -> ValidationResult:
        """Validate a chunk.

        Args:
            chunk: Chunk to validate

        Returns:
            Validation result with scores and issues
        """
        ...

"""Vector math utilities.

Kept intentionally tiny to avoid pulling heavy numeric dependencies (e.g. NumPy)
into the core library.
"""

from __future__ import annotations

import math


def cosine_similarity(
    vec1: list[float],
    vec2: list[float],
    *,
    require_same_length: bool = True,
) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        vec1: First vector.
        vec2: Second vector.
        require_same_length: If True, raise ValueError when vector lengths differ.
            If False, compute using the shared prefix (min length). This is useful
            in best-effort scenarios where callers prefer a safe score over an exception.

    Returns:
        Cosine similarity in [-1, 1]. Returns 0.0 if either vector is empty or has
        zero magnitude.
    """
    if not vec1 or not vec2:
        return 0.0

    if require_same_length and len(vec1) != len(vec2):
        raise ValueError(f"Vector length mismatch: {len(vec1)} != {len(vec2)}")

    n = min(len(vec1), len(vec2))
    dot_product = sum(a * b for a, b in zip(vec1[:n], vec2[:n], strict=True))
    magnitude1 = math.sqrt(sum(a * a for a in vec1[:n]))
    magnitude2 = math.sqrt(sum(b * b for b in vec2[:n]))

    if magnitude1 == 0.0 or magnitude2 == 0.0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)



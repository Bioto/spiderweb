"""Tests for validation components."""

import pytest

from spiderweb.models.document import Chunk, ChunkMetadata, ChunkType
from spiderweb.validators.dedup import DedupValidator
from spiderweb.validators.pipeline import ValidationPipeline
from spiderweb.validators.quality import QualityValidator


def _make_chunk(
    content: str,
    *,
    doc_id: str = "doc",
    idx: int = 0,
    embedding: list[float] | None = None,
) -> Chunk:
    chunk = Chunk(
        content=content,
        metadata=ChunkMetadata(
            document_id=doc_id,
            chunk_index=idx,
            chunk_type=ChunkType.CUSTOM,
        ),
    )
    chunk.embedding = embedding
    return chunk


async def test_quality_validator_fails_low_information_density():
    validator = QualityValidator(min_score=0.3, min_information_density=0.8)
    chunk = _make_chunk("-----.....-----")  # near-zero alnum density

    result = await validator.validate(chunk)

    assert result.passed is False
    assert result.information_density is not None
    assert result.information_density < 0.8


async def test_dedup_validator_detects_exact_duplicates_via_hash():
    validator = DedupValidator(use_content_hash=True, use_embedding_similarity=False)

    chunk1 = _make_chunk("Same content.")
    validator.add_seen(chunk1)

    chunk2 = _make_chunk("Same content.", idx=1)
    result = await validator.validate(chunk2)

    assert result.is_duplicate is True
    assert result.passed is False
    assert "Exact duplicate" in result.issues


async def test_dedup_validator_detects_near_duplicates_via_embedding():
    validator = DedupValidator(use_content_hash=False, use_embedding_similarity=True, threshold=0.95)

    chunk1 = _make_chunk("Chunk A", embedding=[1.0, 0.0])
    validator.add_seen(chunk1)

    chunk2 = _make_chunk("Chunk B", idx=1, embedding=[0.99, 0.01])
    result = await validator.validate(chunk2)

    assert result.is_duplicate is True
    assert result.duplicate_of == chunk1.id


async def test_validation_pipeline_sets_chunk_validation_scores():
    pipeline = ValidationPipeline(llm_client=None)
    chunk = _make_chunk("-----.....-----")

    results = await pipeline.validate_batch([chunk])
    assert len(results) == 1
    assert "quality" in chunk.validation_scores
    assert "information_density" in chunk.validation_scores




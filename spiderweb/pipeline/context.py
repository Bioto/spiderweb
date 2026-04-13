"""Context retrieval for query results.

Retrieves surrounding chunks/pages for query matches with optional
semantic guidance and adaptive expansion.
"""

import asyncio
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from superglue import GlueLLM

from spiderweb.models.config import ContextWindowConfig
from spiderweb.models.document import Chunk
from spiderweb.models.result import ContextChunk, MatchContext
from spiderweb.observability.logging_config import get_logger
from spiderweb.stores.base import VectorStore
from spiderweb.utils.vector_math import cosine_similarity

logger = get_logger(__name__)


class ContextRetriever:
    """Retrieves and optionally scores surrounding context for query matches.
    
    Supports:
    - Page-first or chunk-based context retrieval
    - Semantic guidance for filtering relevant context
    - Adaptive window expansion when initial context is insufficient
    - Deduplication of overlapping context windows
    """

    def __init__(self, vector_store: VectorStore):
        """Initialize context retriever.

        Args:
            vector_store: Vector store to retrieve chunks from
        """
        self.vector_store = vector_store

    def _get_position_field(self, config: ContextWindowConfig) -> str:
        """Get the position field to use based on config.

        Args:
            config: Context window configuration

        Returns:
            Either "chunk_index" or "page_number"
        """
        return "page_number" if config.context_mode == "page" else "chunk_index"

    def _get_reference_position(self, chunk: Chunk, config: ContextWindowConfig) -> int:
        """Get the reference position for a chunk.

        Args:
            chunk: The chunk to get position for
            config: Context window configuration

        Returns:
            Position value (chunk_index or first page_number)
        """
        if config.context_mode == "page":
            # Use first page number, or 0 if none
            return chunk.metadata.page_numbers[0] if chunk.metadata.page_numbers else 0
        else:
            return chunk.metadata.chunk_index

    def _score_chunks(
        self,
        chunks: list[Chunk],
        guide_embedding: list[float],
    ) -> list[tuple[Chunk, float]]:
        """Score chunks against semantic guide embedding.

        Args:
            chunks: Chunks to score
            guide_embedding: Embedding of semantic guide

        Returns:
            List of (chunk, score) tuples
        """
        scored = []
        for chunk in chunks:
            if chunk.embedding:
                score = cosine_similarity(chunk.embedding, guide_embedding, require_same_length=False)
                scored.append((chunk, score))
            else:
                scored.append((chunk, 0.0))

        return scored

    async def _expand_and_retry(
        self,
        document_id: str,
        reference_position: int,
        current_before: int,
        current_after: int,
        config: ContextWindowConfig,
        guide_embedding: list[float],
    ) -> tuple[list[tuple[Chunk, float]], int]:
        """Expand context window if semantic scores are too low.

        Args:
            document_id: Document to search in
            reference_position: Reference position (chunk_index or page_number)
            current_before: Current window before match
            current_after: Current window after match
            config: Context window configuration
            guide_embedding: Semantic guide embedding

        Returns:
            Tuple of (scored_chunks, expansion_steps_used)
        """
        position_field = self._get_position_field(config)
        
        for step in range(config.max_expansion_steps):
            # Expand window by step_size in both directions
            expanded_before = current_before + (step + 1) * config.expansion_step_size
            expanded_after = current_after + (step + 1) * config.expansion_step_size

            # Fetch expanded range
            chunks = await self.vector_store.get_by_position(
                document_id=document_id,
                position_start=max(0, reference_position - expanded_before),
                position_end=reference_position + expanded_after,
                position_field=position_field,
            )

            # Score against semantic guide
            scored = self._score_chunks(chunks, guide_embedding)

            # If any chunk meets threshold, return
            if any(score >= config.semantic_min_score for _, score in scored):
                logger.debug(
                    f"Found relevant context after {step + 1} expansion steps "
                    f"(window: {expanded_before}/{expanded_after})"
                )
                return scored, step + 1

        # Return best effort after max expansions
        logger.debug(f"Reached max expansion steps ({config.max_expansion_steps}), returning best effort")
        return scored, config.max_expansion_steps

    async def _retrieve_context_for_match(
        self,
        match: Chunk,
        match_index: int,
        config: ContextWindowConfig,
        guide_embedding: list[float] | None,
    ) -> MatchContext:
        """Retrieve context for a single match.

        Args:
            match: The matched chunk
            match_index: Index of the match in results
            config: Context window configuration
            guide_embedding: Optional semantic guide embedding

        Returns:
            MatchContext with chunks and metadata
        """
        document_id = match.metadata.document_id
        reference_position = self._get_reference_position(match, config)
        position_field = self._get_position_field(config)

        # Calculate initial range
        position_start = max(0, reference_position - config.chunks_before)
        position_end = reference_position + config.chunks_after

        # Fetch initial context
        chunks = await self.vector_store.get_by_position(
            document_id=document_id,
            position_start=position_start,
            position_end=position_end,
            position_field=position_field,
        )

        expansion_steps = 0
        final_before = config.chunks_before
        final_after = config.chunks_after

        # Apply semantic scoring if guide provided
        if guide_embedding:
            scored_chunks = self._score_chunks(chunks, guide_embedding)

            # Check if we need to expand
            if config.expand_on_low_score:
                max_score = max((score for _, score in scored_chunks), default=0.0)
                
                if max_score < config.semantic_min_score:
                    logger.debug(
                        f"Max semantic score {max_score:.3f} below threshold {config.semantic_min_score}, "
                        f"attempting expansion"
                    )
                    scored_chunks, expansion_steps = await self._expand_and_retry(
                        document_id=document_id,
                        reference_position=reference_position,
                        current_before=config.chunks_before,
                        current_after=config.chunks_after,
                        config=config,
                        guide_embedding=guide_embedding,
                    )
                    
                    # Update final window size
                    final_before = config.chunks_before + expansion_steps * config.expansion_step_size
                    final_after = config.chunks_after + expansion_steps * config.expansion_step_size
        else:
            # No semantic scoring
            scored_chunks = [(chunk, None) for chunk in chunks]

        # Convert to ContextChunk objects
        context_chunks = []
        for chunk, score in scored_chunks:
            # Get this chunk's position (not the match position)
            if config.context_mode == "page":
                # For page mode, prefer page_number from metadata
                chunk_position = chunk.metadata.extra.get("page_number", chunk.metadata.page_numbers[0] if chunk.metadata.page_numbers else chunk.metadata.chunk_index)
            else:
                # For chunk mode, use chunk_index
                chunk_position = chunk.metadata.chunk_index
            
            position_offset = chunk_position - reference_position
            
            context_chunk = ContextChunk(
                chunk={
                    "id": chunk.id,
                    "content": chunk.content,
                    "document_id": chunk.metadata.document_id,
                    "chunk_index": chunk.metadata.chunk_index,
                    "metadata": chunk.metadata.model_dump(),
                },
                position_offset=position_offset,
                semantic_score=score,
                from_expansion=(expansion_steps > 0),
            )
            context_chunks.append(context_chunk)

        # Sort by position offset
        context_chunks.sort(key=lambda c: c.position_offset)

        return MatchContext(
            chunks=context_chunks,
            expansion_steps_used=expansion_steps,
            final_window_size=(final_before, final_after),
        )

    async def get_context_for_matches(
        self,
        matches: list[Chunk],
        config: ContextWindowConfig,
        llm_client: "GlueLLM | None" = None,
    ) -> dict[int, MatchContext]:
        """Get context chunks for each match, with optional semantic scoring.

        Args:
            matches: List of matched chunks
            config: Context window configuration
            llm_client: Optional LLM client for semantic guide embedding

        Returns:
            Dictionary mapping match index to MatchContext
        """
        if not config.enabled:
            logger.debug("Context window disabled, returning empty context")
            return {}

        if not matches:
            return {}

        # Generate guide embedding if provided
        guide_embedding = None
        if config.semantic_guide and llm_client:
            logger.debug(f"Generating embedding for semantic guide: {config.semantic_guide[:50]}...")
            try:
                embedding_result = await llm_client.embed(config.semantic_guide)
                guide_embedding = embedding_result.embeddings[0]
            except Exception as e:
                logger.warning(f"Failed to generate semantic guide embedding: {e}. Continuing without semantic scoring.")

        # Group matches by document_id for efficient batching
        matches_by_doc = defaultdict(list)
        for idx, match in enumerate(matches):
            matches_by_doc[match.metadata.document_id].append((idx, match))

        logger.info(
            f"Retrieving context for {len(matches)} matches across {len(matches_by_doc)} documents "
            f"(mode={config.context_mode}, window={config.chunks_before}/{config.chunks_after})"
        )

        # Process matches concurrently
        tasks = []
        match_indices = []
        
        for doc_id, doc_matches in matches_by_doc.items():
            for match_idx, match in doc_matches:
                tasks.append(
                    self._retrieve_context_for_match(
                        match=match,
                        match_index=match_idx,
                        config=config,
                        guide_embedding=guide_embedding,
                    )
                )
                match_indices.append(match_idx)

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Build result dictionary
        context_by_match = {}
        for match_idx, result in zip(match_indices, results, strict=True):
            if isinstance(result, Exception):
                logger.error(f"Failed to retrieve context for match {match_idx}: {result}")
                # Return empty context for failed match
                context_by_match[match_idx] = MatchContext(
                    chunks=[],
                    expansion_steps_used=0,
                    final_window_size=(0, 0),
                )
            else:
                context_by_match[match_idx] = result

        logger.info(f"Successfully retrieved context for {len(context_by_match)} matches")
        
        return context_by_match


"""Re-ranking for improved search precision.

Re-ranks vector search results using cross-encoders or API-based reranking
services to improve precision by scoring query-document relevance more accurately.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gluellm import GlueLLM

    from spiderweb.models.document import Chunk

from spiderweb.observability.logging_config import get_logger

logger = get_logger(__name__)


class Reranker:
    """Re-ranks search results for improved precision.

    After initial vector search, re-ranks results using a more accurate
    relevance model (cross-encoder or API-based reranking service).

    Example:
        >>> reranker = Reranker(llm_client=llm, model="cross-encoder")
        >>> reranked = await reranker.rerank(query, results, top_k=5)
    """

    def __init__(
        self,
        llm_client: "GlueLLM | None" = None,
        model: str = "cross-encoder",
        **kwargs,
    ):
        """Initialize reranker.

        Args:
            llm_client: GlueLLM client (for API-based reranking)
            model: Reranking model type ("cross-encoder", "cohere", or "none")
            **kwargs: Additional model-specific options
        """
        self.llm_client = llm_client
        self.model = model
        self._kwargs = kwargs

    async def rerank(
        self,
        query: str,
        results: list[tuple["Chunk", float]],
        top_k: int | None = None,
    ) -> list[tuple["Chunk", float]]:
        """Re-rank search results.

        Args:
            query: Original search query
            results: List of (chunk, score) tuples from vector search
            top_k: Number of top results to return (None = return all)

        Returns:
            Re-ranked list of (chunk, score) tuples
        """
        if not results:
            return []

        if self.model == "none":
            # No reranking, return as-is
            return results[:top_k] if top_k else results

        if self.model == "cross-encoder":
            return await self._rerank_cross_encoder(query, results, top_k)
        elif self.model == "cohere":
            return await self._rerank_cohere(query, results, top_k)
        else:
            logger.warning(f"Unknown reranking model: {self.model}, skipping reranking")
            return results[:top_k] if top_k else results

    async def _rerank_cross_encoder(
        self,
        query: str,
        results: list[tuple["Chunk", float]],
        top_k: int | None,
    ) -> list[tuple["Chunk", float]]:
        """Re-rank using a cross-encoder model.

        Uses sentence-transformers cross-encoder for accurate relevance scoring.

        Args:
            query: Search query
            results: Initial results
            top_k: Number of results to return

        Returns:
            Re-ranked results
        """
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. Install with: pip install sentence-transformers. "
                "Skipping cross-encoder reranking."
            )
            return results[:top_k] if top_k else results

        model_name = self._kwargs.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        cross_encoder = CrossEncoder(model_name)

        # Prepare query-document pairs
        pairs = [(query, chunk.content) for chunk, _ in results]

        # Score pairs
        scores = cross_encoder.predict(pairs)

        # Combine with original results and sort
        reranked = [
            (chunk, float(score))
            for (chunk, _), score in zip(results, scores)
        ]
        reranked.sort(key=lambda x: x[1], reverse=True)

        logger.debug(f"Re-ranked {len(results)} results using cross-encoder")
        return reranked[:top_k] if top_k else reranked

    async def _rerank_cohere(
        self,
        query: str,
        results: list[tuple["Chunk", float]],
        top_k: int | None,
    ) -> list[tuple["Chunk", float]]:
        """Re-rank using Cohere rerank API.

        Args:
            query: Search query
            results: Initial results
            top_k: Number of results to return

        Returns:
            Re-ranked results
        """
        if not self.llm_client:
            logger.warning("LLM client required for Cohere reranking, skipping")
            return results[:top_k] if top_k else results

        try:
            import cohere
        except ImportError:
            logger.warning(
                "cohere not installed. Install with: pip install cohere. "
                "Skipping Cohere reranking."
            )
            return results[:top_k] if top_k else results

        # Get Cohere API key from environment or llm_client config
        import os

        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            logger.warning("COHERE_API_KEY not set, skipping Cohere reranking")
            return results[:top_k] if top_k else results

        client = cohere.Client(api_key=api_key)

        # Prepare documents
        documents = [chunk.content for chunk, _ in results]

        # Rerank
        rerank_response = client.rerank(
            model=self._kwargs.get("model", "rerank-english-v3.0"),
            query=query,
            documents=documents,
            top_n=top_k or len(results),
        )

        # Map back to chunks
        reranked = []
        for result in rerank_response.results:
            idx = result.index
            chunk, _ = results[idx]
            reranked.append((chunk, float(result.relevance_score)))

        logger.debug(f"Re-ranked {len(results)} results using Cohere")
        return reranked

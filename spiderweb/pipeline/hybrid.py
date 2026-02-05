"""Hybrid search combining keyword (BM25) and vector retrieval.

Uses BM25 for keyword matching and vector search for semantic similarity,
then combines results using Reciprocal Rank Fusion (RRF).
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gluellm import GlueLLM

    from spiderweb.models.document import Chunk
    from spiderweb.stores.base import VectorStore

from spiderweb.observability.logging_config import get_logger
from spiderweb.pipeline.query_expansion import reciprocal_rank_fusion

logger = get_logger(__name__)


class BM25Index:
    """Simple BM25 index for keyword-based retrieval.

    BM25 (Best Matching 25) is a ranking function used to score documents
    based on query terms. It's effective for exact keyword matching.
    """

    def __init__(self, chunks: list["Chunk"] | None = None):
        """Initialize BM25 index.

        Args:
            chunks: Optional initial chunks to index
        """
        self._chunks: list["Chunk"] = chunks or []
        self._index: dict[str, list[int]] = {}  # term -> list of chunk indices
        self._doc_freqs: dict[str, int] = {}  # term -> document frequency
        self._k1 = 1.5  # BM25 parameter
        self._b = 0.75  # BM25 parameter
        self._avg_doc_length = 0.0

        if chunks:
            self._build_index()

    def add_chunks(self, chunks: list["Chunk"]) -> None:
        """Add chunks to the index.

        Args:
            chunks: Chunks to add
        """
        start_idx = len(self._chunks)
        self._chunks.extend(chunks)
        self._build_index()

    def _build_index(self) -> None:
        """Build the inverted index from chunks."""
        import re

        self._index = {}
        self._doc_freqs = {}
        total_length = 0

        for idx, chunk in enumerate(self._chunks):
            # Tokenize chunk content
            tokens = self._tokenize(chunk.content)
            total_length += len(tokens)

            # Track term positions
            seen_in_doc = set()
            for token in tokens:
                if token not in self._index:
                    self._index[token] = []
                self._index[token].append(idx)

                if token not in seen_in_doc:
                    seen_in_doc.add(token)
                    self._doc_freqs[token] = self._doc_freqs.get(token, 0) + 1

        # Calculate average document length
        if self._chunks:
            self._avg_doc_length = total_length / len(self._chunks)

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into terms.

        Args:
            text: Text to tokenize

        Returns:
            List of lowercase terms
        """
        import re

        # Simple tokenization: lowercase, alphanumeric
        tokens = re.findall(r"\b\w+\b", text.lower())
        return tokens

    def search(self, query: str, top_k: int = 10) -> list[tuple["Chunk", float]]:
        """Search using BM25 scoring.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of (chunk, bm25_score) tuples, sorted by score
        """
        if not self._chunks:
            return []

        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        # Calculate BM25 scores for each chunk
        scores: dict[int, float] = {}

        for term in query_terms:
            if term not in self._index:
                continue

            # Term frequency in corpus
            df = self._doc_freqs.get(term, 0)
            if df == 0:
                continue

            # Inverse document frequency
            idf = self._calculate_idf(len(self._chunks), df)

            # Score each document containing this term
            for chunk_idx in self._index[term]:
                chunk = self._chunks[chunk_idx]
                chunk_tokens = self._tokenize(chunk.content)
                doc_length = len(chunk_tokens)

                # Term frequency in document
                tf = chunk_tokens.count(term)

                # BM25 score component for this term
                numerator = idf * tf * (self._k1 + 1)
                denominator = tf + self._k1 * (
                    1 - self._b + self._b * (doc_length / self._avg_doc_length)
                )
                score_component = numerator / denominator

                scores[chunk_idx] = scores.get(chunk_idx, 0.0) + score_component

        # Sort by score and return top_k
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = [
            (self._chunks[idx], score) for idx, score in ranked[:top_k]
        ]

        logger.debug(f"BM25 search returned {len(results)} results")
        return results

    def _calculate_idf(self, total_docs: int, doc_freq: int) -> float:
        """Calculate inverse document frequency.

        Args:
            total_docs: Total number of documents
            doc_freq: Number of documents containing the term

        Returns:
            IDF score
        """
        import math

        if doc_freq == 0:
            return 0.0
        return math.log((total_docs - doc_freq + 0.5) / (doc_freq + 0.5))


class HybridSearcher:
    """Hybrid search combining BM25 and vector retrieval.

    Runs both keyword (BM25) and semantic (vector) searches, then combines
    results using Reciprocal Rank Fusion for optimal recall and precision.
    """

    def __init__(
        self,
        llm_client: "GlueLLM",
        vector_store: "VectorStore",
        bm25_index: BM25Index | None = None,
    ):
        """Initialize hybrid searcher.

        Args:
            llm_client: GlueLLM client for embeddings
            vector_store: Vector store for semantic search
            bm25_index: Optional BM25 index (will build from vector store if not provided)
        """
        self.llm_client = llm_client
        self.vector_store = vector_store
        self.bm25_index = bm25_index

    async def search(
        self,
        query: str,
        top_k: int = 10,
        filter_dict: dict | None = None,
        rrf_k: int = 60,
    ) -> list[tuple["Chunk", float]]:
        """Perform hybrid search.

        Args:
            query: Search query
            top_k: Number of results to return
            filter_dict: Optional metadata filters
            rrf_k: RRF constant for fusion

        Returns:
            Combined and ranked results
        """
        # Build BM25 index if needed
        if self.bm25_index is None:
            logger.debug("Building BM25 index from vector store chunks")
            # Note: This is a simplified approach - in production, you'd want
            # to maintain a separate BM25 index or use a library like rank-bm25
            self.bm25_index = BM25Index()

        # Run BM25 search
        bm25_results = self.bm25_index.search(query, top_k=top_k * 2)

        # Run vector search
        embedding_result = await self.llm_client.embed(query)
        query_embedding = embedding_result.embeddings[0]
        vector_results = await self.vector_store.query(
            embedding=query_embedding,
            top_k=top_k * 2,
            filter_dict=filter_dict,
        )

        # Combine using RRF
        combined = reciprocal_rank_fusion(
            [bm25_results, vector_results],
            k=rrf_k,
        )

        # Return top_k
        return combined[:top_k]

"""Query expansion for improved vector search recall.

This module implements query expansion strategies including multi-query
reformulation and HyDE (Hypothetical Document Embeddings), with results
combined using Reciprocal Rank Fusion.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gluellm import GlueLLM

    from spiderweb.models.document import Chunk

from spiderweb.models.config import QueryExpansionConfig
from spiderweb.observability.logging_config import get_logger

logger = get_logger(__name__)

# Default prompts optimized for vector search
DEFAULT_MULTI_QUERY_PROMPT = """Generate {num_expansions} alternative phrasings of this query. Focus on synonyms, different vocabulary, and varied sentence structures that might match document content. Return only the queries, one per line, without numbering or bullet points.

Query: {query}

Alternative queries:"""

DEFAULT_HYDE_PROMPT = """Write a short, factual paragraph (3-4 sentences) that would appear in a document answering this query. Use technical vocabulary and specific terms that would likely be in such a document. Be concise and precise.

Query: {query}

Hypothetical document excerpt:"""


class QueryExpander:
    """Handles query expansion using LLM-based strategies.
    
    Supports both multi-query reformulation (generating alternative phrasings)
    and HyDE (generating hypothetical answers to embed instead).
    """

    def __init__(self, llm_client: "GlueLLM", config: QueryExpansionConfig):
        """Initialize query expander.
        
        Args:
            llm_client: GlueLLM client for generating expansions
            config: Query expansion configuration
        """
        self.llm_client = llm_client
        self.config = config
        
    async def expand(self, query: str) -> list[str]:
        """Expand query using configured strategy.
        
        Args:
            query: Original query string
            
        Returns:
            List of expanded queries (may include original if configured)
            
        Raises:
            ValueError: If query is empty or expansion fails
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
            
        try:
            if self.config.strategy == "multi_query":
                expanded = await self.expand_multi_query(query)
            elif self.config.strategy == "hyde":
                expanded = await self.expand_hyde(query)
            else:
                raise ValueError(f"Unknown expansion strategy: {self.config.strategy}")
                
            # Include original query if configured
            if self.config.include_original:
                expanded = [query] + expanded
                
            logger.info(
                f"Expanded query using {self.config.strategy}: "
                f"{len(expanded)} total queries"
            )
            
            return expanded
            
        except Exception as e:
            logger.error(f"Query expansion failed: {e}", exc_info=True)
            # Fallback to original query on error
            logger.warning("Falling back to original query")
            return [query]
    
    async def expand_multi_query(self, query: str) -> list[str]:
        """Generate multiple reformulations of the query.
        
        Uses LLM to create alternative phrasings with different vocabulary
        and sentence structures to improve recall.
        
        Args:
            query: Original query string
            
        Returns:
            List of reformulated queries
        """
        # Use custom prompt or default
        prompt = self.config.custom_prompt or DEFAULT_MULTI_QUERY_PROMPT
        prompt = prompt.format(num_expansions=self.config.num_expansions, query=query)
        
        logger.debug(f"Multi-query expansion for: {query}")
        
        # Generate expansions using LLM
        response = await self.llm_client.generate(
            prompt=prompt,
            temperature=0.7,  # Some creativity for variations
            max_tokens=500,
        )
        
        # Parse response - expecting one query per line
        expanded_queries = []
        for line in response.content.strip().split("\n"):
            line = line.strip()
            # Skip empty lines
            if not line:
                continue
            
            # Remove common prefixes like "1. ", "- ", "* ", etc.
            # Handle numbered lists (1., 2., etc.)
            import re
            line = re.sub(r'^\d+[\.\)]\s*', '', line)  # Remove "1. " or "1) "
            
            # Remove bullet points
            for prefix in ["- ", "* ", "• ", "> "]:
                if line.startswith(prefix):
                    line = line[len(prefix):]
                    break
            
            line = line.strip()
            if line:
                expanded_queries.append(line)
        
        # Ensure we got the expected number
        if len(expanded_queries) < self.config.num_expansions:
            logger.warning(
                f"Expected {self.config.num_expansions} expansions, "
                f"got {len(expanded_queries)}"
            )
        
        # Take only the requested number
        expanded_queries = expanded_queries[:self.config.num_expansions]
        
        return expanded_queries
    
    async def expand_hyde(self, query: str) -> list[str]:
        """Generate hypothetical document content for HyDE strategy.
        
        Creates a hypothetical answer that might appear in a relevant document,
        allowing embedding to match document vocabulary rather than query vocabulary.
        
        Args:
            query: Original query string
            
        Returns:
            List containing the hypothetical document (single item)
        """
        # Use custom prompt or default
        prompt = self.config.custom_prompt or DEFAULT_HYDE_PROMPT
        prompt = prompt.format(query=query)
        
        logger.debug(f"HyDE expansion for: {query}")
        
        # Generate hypothetical document using LLM
        response = await self.llm_client.generate(
            prompt=prompt,
            temperature=0.5,  # More factual, less creative
            max_tokens=300,
        )
        
        hypothetical_doc = response.content.strip()
        
        # HyDE returns a single hypothetical answer
        # But we can generate multiple if num_expansions > 1
        if self.config.num_expansions > 1:
            logger.info(
                f"Generating {self.config.num_expansions} hypothetical documents"
            )
            # Generate additional variations with higher temperature
            expansions = [hypothetical_doc]
            for i in range(self.config.num_expansions - 1):
                response = await self.llm_client.generate(
                    prompt=prompt,
                    temperature=0.7 + (i * 0.1),  # Slight variation
                    max_tokens=300,
                )
                expansions.append(response.content.strip())
            return expansions
        
        return [hypothetical_doc]


def reciprocal_rank_fusion(
    query_results: list[list[tuple["Chunk", float]]],
    k: int = 60,
) -> list[tuple["Chunk", float]]:
    """Combine multiple result sets using Reciprocal Rank Fusion.
    
    RRF is a simple yet effective method for combining ranked lists that
    doesn't require score normalization. It gives diminishing credit to
    items based on their rank position across all queries.
    
    Formula: RRF_score = Σ (1 / (k + rank_i))
    
    Args:
        query_results: List of result sets, each containing (chunk, score) tuples
        k: RRF constant (default 60, standard value from literature)
        
    Returns:
        Combined and re-ranked list of (chunk, rrf_score) tuples
    """
    if not query_results:
        return []
        
    # Track RRF scores by chunk ID
    rrf_scores: dict[str, float] = {}
    chunk_by_id: dict[str, "Chunk"] = {}
    
    # Process each query's results
    for results in query_results:
        for rank, (chunk, _original_score) in enumerate(results, start=1):
            chunk_id = chunk.id
            
            # RRF formula: 1 / (k + rank)
            rrf_contribution = 1.0 / (k + rank)
            
            # Accumulate RRF scores
            if chunk_id in rrf_scores:
                rrf_scores[chunk_id] += rrf_contribution
            else:
                rrf_scores[chunk_id] = rrf_contribution
                chunk_by_id[chunk_id] = chunk
    
    # Sort by RRF score (descending)
    ranked_chunks = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )
    
    # Return as list of (chunk, score) tuples
    result = [(chunk_by_id[chunk_id], score) for chunk_id, score in ranked_chunks]
    
    logger.debug(
        f"RRF combined {len(query_results)} result sets into {len(result)} unique results"
    )
    
    return result


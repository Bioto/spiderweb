"""Crawl relevance filtering using LLM batching.

Filters URLs/links based on a prompt describing what is good vs bad
to crawl, using GlueLLM's batching feature for efficiency.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gluellm import GlueLLM

from spiderweb.observability.logging_config import get_logger
from spiderweb.search.base import SearchResult

logger = get_logger(__name__)


async def filter_by_relevance(
    llm_client: "GlueLLM | None",
    candidates: list[SearchResult | dict[str, Any]],
    relevance_prompt: str,
    use_llm: bool = True,
) -> tuple[list[SearchResult | dict[str, Any]], list[SearchResult | dict[str, Any]]]:
    """Filter candidates by relevance using LLM batching.
    
    Uses GlueLLM's batching feature to evaluate multiple candidates
    in one or few calls, classifying each as "good" or "bad" to crawl.
    
    Args:
        llm_client: GlueLLM client (required if use_llm=True)
        candidates: List of SearchResult objects or dicts with url/title/snippet
        relevance_prompt: Prompt describing what is good vs bad to crawl
        use_llm: If True, use LLM; if False, fall back to keyword heuristic
        
    Returns:
        Tuple of (good_candidates, bad_candidates) where bad_candidates
        have filter_reason set
        
    Raises:
        ValueError: If use_llm=True but llm_client is None
    """
    if not candidates:
        return [], []
    
    if not use_llm or llm_client is None:
        # Fallback: simple keyword-based heuristic
        logger.debug("Using keyword-based relevance filter (LLM not available)")
        return _filter_by_keywords(candidates, relevance_prompt)
    
    logger.debug(f"Filtering {len(candidates)} candidates using LLM batching")
    
    # Build batch of prompts for all candidates
    batch_prompts = []
    for candidate in candidates:
        url = candidate.url if isinstance(candidate, SearchResult) else candidate.get("url", "")
        title = (
            candidate.title
            if isinstance(candidate, SearchResult)
            else candidate.get("title", "")
        )
        snippet = (
            candidate.description or candidate.snippet
            if isinstance(candidate, SearchResult)
            else candidate.get("snippet") or candidate.get("description", "")
        )
        
        prompt = _build_relevance_prompt(relevance_prompt, url, title, snippet)
        batch_prompts.append(prompt)
    
    # Use GlueLLM batching to evaluate all at once
    try:
        # GlueLLM's generate_batch or similar batching API
        # Check if it has a batch method, otherwise fall back to individual calls
        if hasattr(llm_client, "generate_batch"):
            responses = await llm_client.generate_batch(batch_prompts, temperature=0.0)
        elif hasattr(llm_client, "batch_generate"):
            responses = await llm_client.batch_generate(batch_prompts, temperature=0.0)
        else:
            # Fallback: use asyncio.gather for concurrent individual calls
            import asyncio
            
            async def single_generate(prompt: str) -> Any:
                result = await llm_client.generate(prompt, temperature=0.0)
                return result.text if hasattr(result, "text") else str(result)
            
            responses = await asyncio.gather(*[single_generate(p) for p in batch_prompts])
            # Normalize responses to have .text attribute
            class Response:
                def __init__(self, text: str):
                    self.text = text
            
            responses = [Response(r) if isinstance(r, str) else r for r in responses]
        
        # Parse responses and classify
        good_candidates = []
        bad_candidates = []
        
        for candidate, response in zip(candidates, responses, strict=True):
            response_text = response.text if hasattr(response, "text") else str(response)
            is_good = _parse_relevance_response(response_text)
            
            if is_good:
                good_candidates.append(candidate)
            else:
                # Create filtered candidate with reason
                if isinstance(candidate, SearchResult):
                    from spiderweb.search.trace import FilteredCandidate
                    
                    bad_candidates.append(
                        FilteredCandidate(
                            url=candidate.url,
                            title=candidate.title,
                            snippet=candidate.description or candidate.snippet,
                            filter_reason="relevance",
                        )
                    )
                else:
                    # Dict case - convert to FilteredCandidate-like dict
                    bad_candidate = dict(candidate)
                    bad_candidate["filtered_out"] = True
                    bad_candidate["filter_reason"] = "relevance"
                    bad_candidates.append(bad_candidate)
        
        logger.info(f"Filtered {len(candidates)} candidates: {len(good_candidates)} good, {len(bad_candidates)} bad")
        return good_candidates, bad_candidates
    
    except Exception as e:
        logger.warning(f"LLM relevance filtering failed: {e}. Falling back to keyword heuristic.")
        return _filter_by_keywords(candidates, relevance_prompt)


def _build_relevance_prompt(relevance_prompt: str, url: str, title: str, snippet: str) -> str:
    """Build a prompt for evaluating a single candidate.
    
    Args:
        relevance_prompt: User's description of good vs bad
        url: Candidate URL
        title: Candidate title
        snippet: Candidate description/snippet
        
    Returns:
        Complete prompt for LLM
    """
    return f"""You are evaluating whether a URL should be crawled based on relevance criteria.

Relevance criteria:
{relevance_prompt}

Candidate to evaluate:
- URL: {url}
- Title: {title}
- Description: {snippet or "(no description)"}

Respond with only "GOOD" if this URL should be crawled, or "BAD" if it should be skipped.
Response:"""


def _parse_relevance_response(response_text: str) -> bool:
    """Parse LLM response to determine if candidate is good.
    
    Args:
        response_text: LLM response text
        
    Returns:
        True if candidate is good (should crawl), False if bad (skip)
    """
    response_lower = response_text.strip().upper()
    # Look for explicit GOOD/BAD indicators
    if "GOOD" in response_lower and "BAD" not in response_lower:
        return True
    if "BAD" in response_lower:
        return False
    # Default: if unclear, assume good (don't filter out)
    return True


def _filter_by_keywords(
    candidates: list[SearchResult | dict[str, Any]],
    relevance_prompt: str,
) -> tuple[list[SearchResult | dict[str, Any]], list[SearchResult | dict[str, Any]]]:
    """Simple keyword-based filtering fallback.
    
    Extracts "Bad:" keywords from prompt and filters candidates
    that contain those keywords in URL/title/snippet.
    
    Args:
        candidates: List of candidates to filter
        relevance_prompt: Prompt with "Bad:" section
        
    Returns:
        Tuple of (good_candidates, bad_candidates)
    """
    # Extract bad keywords from prompt
    bad_keywords = []
    if "Bad:" in relevance_prompt:
        bad_section = relevance_prompt.split("Bad:")[1].split("\n")[0]
        bad_keywords = [kw.strip().lower() for kw in bad_section.split(",")]
    
    good_candidates = []
    bad_candidates = []
    
    for candidate in candidates:
        url = candidate.url if isinstance(candidate, SearchResult) else candidate.get("url", "")
        title = (
            candidate.title
            if isinstance(candidate, SearchResult)
            else candidate.get("title", "")
        )
        snippet = (
            candidate.description or candidate.snippet
            if isinstance(candidate, SearchResult)
            else candidate.get("snippet") or candidate.get("description", "")
        )
        
        text_to_check = f"{url} {title} {snippet}".lower()
        
        # Check if any bad keyword appears
        is_bad = any(keyword in text_to_check for keyword in bad_keywords if keyword)
        
        if is_bad:
            if isinstance(candidate, SearchResult):
                from spiderweb.search.trace import FilteredCandidate
                
                bad_candidates.append(
                    FilteredCandidate(
                        url=candidate.url,
                        title=candidate.title,
                        snippet=candidate.description or candidate.snippet,
                        filter_reason="relevance (keyword)",
                    )
                )
            else:
                bad_candidate = dict(candidate)
                bad_candidate["filtered_out"] = True
                bad_candidate["filter_reason"] = "relevance (keyword)"
                bad_candidates.append(bad_candidate)
        else:
            good_candidates.append(candidate)
    
    return good_candidates, bad_candidates

"""Build X (Twitter) Search API query strings from natural-language search terms.

Uses an LLM to turn a user query (e.g. "recent posts about MSP from verified users")
into a valid X Search query using operators from:
https://docs.x.com/x-api/posts/search/integrate/operators
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

from spiderweb.observability.logging_config import get_logger

if TYPE_CHECKING:
    from gluellm import GlueLLM

logger = get_logger(__name__)

# Concise reference for the LLM: X Search API operators (standalone + conjunction-required).
# Query limits: recent 512 chars, full-archive 1024 (self-serve).
X_SEARCH_OPERATORS_REF = """
X Search query operators (use only these; output a single query string):

Keywords/phrases: keyword (tokenized), "exact phrase", emoji.
Entities: #hashtag (exact), @username (mention), $cashtag.
Users: from:username, to:username, retweets_of:username.
URLs: url:"https://..."
Context: context:DOMAIN_ENTITY_ID, entity:"string" (recent only), conversation_id:ID.
List: list:LIST_ID.
Post refs: in_reply_to_tweet_id:ID, retweets_of_tweet_id:ID, quotes_of_tweet_id:ID.
Location: place:"name", place_country:CC, point_radius:[lon lat radius], bounding_box:[...].
Post type (conjunction-required): is:retweet, is:reply, is:quote, is:verified, -is:nullcast.
Content (conjunction-required): has:hashtags, has:cashtags, has:links, has:mentions, has:media, has:images, has:video_link, has:geo.
Language (conjunction-required): lang:en (BCP 47 codes: en, es, fr, de, ja, etc.).

Logic: OR, space=AND, () for grouping, - for negation.
Example: (python OR #python) -is:retweet lang:en
"""


async def build_x_search_query(user_query: str, llm_client: GlueLLM) -> str:
    """Turn a natural-language or short search term into an X Search API query string.

    The LLM is prompted with the X Search operators reference and asked to return
    only a single line: the query string to pass to the Search API. Useful when
    the user says e.g. "msp" or "recent verified posts about Python" and you want
    a richer query (hashtags, filters, language, etc.).

    Args:
        user_query: Raw search term or natural-language description.
        llm_client: GlueLLM client for completion.

    Returns:
        A single-line X Search query string (stripped; no markdown or explanation).
    """
    prompt = (
        "You are a search expert. Interpret what the user is trying to find on X (Twitter) and output "
        "exactly one X Search API query string that will retrieve that. Never use the user's input verbatim: "
        "always translate their intent into a proper search query (keywords, phrases, hashtags, filters) using only "
        "the operators below. Output nothing else: no explanation, no markdown, no quotes around the whole line—just "
        "the query. Keep under 512 characters. Optionally include relevant hashtags (e.g. #topic) when they would "
        "improve recall; use the # operator for exact hashtag match, e.g. (keyword OR #hashtag).\n\n"
        f"{X_SEARCH_OPERATORS_REF.strip()}\n\n"
        f"User search intent: {user_query}\n\n"
        "Output the single X search query line (adjusted for what they are actually trying to get):"
    )
    if not hasattr(llm_client, "complete"):
        raise ValueError("llm_client must provide complete(user_message=..., ...)")
    kwargs: dict = {"user_message": prompt}
    sig = inspect.signature(llm_client.complete)
    if "model_kwargs" in sig.parameters:
        kwargs["model_kwargs"] = {"temperature": 0.3, "max_tokens": 256}
    response = await llm_client.complete(**kwargs)

    if hasattr(response, "final_response"):
        content = (response.final_response or "").strip()
    elif hasattr(response, "content"):
        content = (response.content or "").strip()
    else:
        content = str(response).strip()

    # Strip markdown code fence if present
    if content.startswith("```"):
        lines = content.split("\n")
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()
    # Take first line only (query is one line)
    query_line = content.split("\n")[0].strip()
    if not query_line:
        logger.warning("X query builder returned empty; using original user query")
        return user_query.strip()
    return query_line[:512]

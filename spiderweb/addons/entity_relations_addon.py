"""Relationship extraction add-on: entity–entity relations from document and entities.

Runs after chunking; reads document text and optional entity list (e.g. from
document.metadata.extra["langextract"]) and uses an LLM to extract relationship
triples. Writes document.metadata.extra["relationships"] for the pipeline to
push to the graph store.

Enable with ChunkAddOnConfig(enabled=["entity_entity_relations"], ...).
Run after LangExtract so entity ids match (same entity_id_from_extraction convention).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from spiderweb.addons.base import ChunkAddOn
from spiderweb.config import settings
from spiderweb.observability.logging_config import get_logger
from spiderweb.pipeline.graph_adapter import entity_id_from_extraction

if TYPE_CHECKING:
    from spiderweb.models.document import Chunk, Document

logger = get_logger(__name__)


class RelationTriple(BaseModel):
    """Single (source, relation_type, target) triple."""

    source: str = Field(description="Source entity label or id")
    relation_type: str = Field(description="Type of relationship (e.g. WORKS_AT, KNOWS)")
    target: str = Field(description="Target entity label or id")


class RelationshipsResponse(BaseModel):
    """LLM output for relationship extraction."""

    relationships: list[RelationTriple] = Field(
        description="List of (source, relation_type, target) triples",
        default_factory=list,
    )


class EntityEntityRelationsAddOn:
    """Add-on that extracts entity–entity relationships from document content.

    Uses document text and optional entities from document.metadata.extra["langextract"]
    to produce triples written to document.metadata.extra["relationships"].
    When LangExtract extractions exist, maps entity labels to the same stable entity ids
    used by the graph adapter so relationships reference the correct nodes.
    """

    def __init__(
        self,
        llm_client: Any | None = None,
        model: str | None = None,
        max_relations: int = 50,
        max_text_chars: int = 12_000,
        **kwargs: object,
    ) -> None:
        self.llm_client = llm_client
        self.model = model
        self.max_relations = max_relations
        self.max_text_chars = max_text_chars

    async def process_async(
        self,
        chunks: "list[Chunk]",
        *,
        document: "Document | None" = None,
        **kwargs: object,
    ) -> "list[Chunk]":
        """Extract relationship triples from document and write to document.metadata.extra["relationships"]."""
        if not document:
            logger.debug("EntityEntityRelationsAddOn: no document provided, skipping")
            return chunks

        if not self.llm_client:
            logger.warning("EntityEntityRelationsAddOn: No LLM client provided, skipping")
            return chunks

        text = document.markdown_content or document.raw_content or ""
        if not text.strip():
            logger.debug("EntityEntityRelationsAddOn: document has no content, skipping")
            self._set_relationships(document, [])
            return chunks

        extra = document.metadata.extra or {}
        langextract = extra.get("langextract") or {}
        extractions: list[dict[str, Any]] = []
        if isinstance(langextract, dict):
            extractions = list(langextract.get("extractions") or [])

        # Label -> entity id (same ids as graph_adapter uses)
        label_to_id: dict[str, str] = {}
        for i, ex in enumerate(extractions):
            if not isinstance(ex, dict):
                continue
            eid = entity_id_from_extraction(document.id, i, ex)
            label = (ex.get("extraction_text") or "").strip()
            if label:
                label_to_id[label] = eid
                # Normalize case and strip for matching
                if label.lower() not in label_to_id:
                    label_to_id[label.lower()] = eid

        def resolve_id(label_or_id: str) -> str:
            s = (label_or_id or "").strip()
            if not s:
                return ""
            if s in label_to_id:
                return label_to_id[s]
            if s.lower() in label_to_id:
                return label_to_id[s.lower()]
            return s

        snippet = text[: self.max_text_chars] if len(text) > self.max_text_chars else text
        if len(text) > self.max_text_chars:
            snippet += "\n[... truncated ...]"

        entity_list = list(label_to_id.keys())[:200]
        entities_desc = (
            f"Known entities (use these exact labels as source/target): {entity_list!r}"
            if entity_list
            else "No predefined entity list; extract (subject, relation, object) from the text and use consistent labels."
        )

        prompt = f"""Extract relationship triples from the following document.

{entities_desc}

Document excerpt:
{snippet}

For each relationship, output the source entity (label or id), the relation type (e.g. WORKS_AT, LOCATED_IN, KNOWS, PART_OF), and the target entity.
Return a JSON object with a "relationships" array: [{{"source": "...", "relation_type": "...", "target": "..."}}].
Limit to {self.max_relations} triples. Use consistent labels that match the entity list when provided."""

        try:
            try:
                from gluellm.api import structured_complete

                response = await structured_complete(
                    user_message=prompt,
                    response_format=RelationshipsResponse,
                    model=self.model or "default",
                    timeout=settings.llm_timeout,
                )
                triples = response.relationships[: self.max_relations] if response.relationships else []
            except (ImportError, AttributeError, Exception):
                triples = await self._extract_via_complete(prompt)
        except Exception as e:
            logger.warning("EntityEntityRelationsAddOn: LLM extraction failed: %s", e)
            self._set_relationships(document, [])
            return chunks

        relationships: list[dict[str, Any]] = []
        for t in triples:
            sid = resolve_id(t.source)
            tid = resolve_id(t.target)
            if not sid or not tid or not (t.relation_type or "").strip():
                continue
            relationships.append({
                "source_id": sid,
                "target_id": tid,
                "relation_type": (t.relation_type or "").strip(),
                "attributes": {},
            })

        self._set_relationships(document, relationships)
        logger.info(
            "EntityEntityRelationsAddOn: extracted %d relationships for document %s",
            len(relationships),
            document.id[:8],
        )
        return chunks

    async def _extract_via_complete(self, prompt: str) -> list[RelationTriple]:
        """Fallback: use complete() and parse JSON."""
        complete_kwargs: dict[str, Any] = {
            "user_message": prompt,
            "temperature": 0.2,
            "timeout": settings.llm_timeout,
        }
        if self.model is not None:
            complete_kwargs["model"] = self.model

        try:
            response = await self.llm_client.complete(**complete_kwargs)
        except TypeError:
            complete_kwargs.pop("temperature", None)
            complete_kwargs.pop("model", None)
            complete_kwargs.pop("timeout", None)
            response = await self.llm_client.complete(**complete_kwargs)

        content = getattr(response, "final_response", None) or getattr(response, "content", None) or str(response).strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        try:
            data = json.loads(content)
            if isinstance(data, dict) and "relationships" in data:
                raw = data["relationships"]
                if isinstance(raw, list):
                    out: list[RelationTriple] = []
                    for r in raw[: self.max_relations]:
                        if not isinstance(r, dict):
                            continue
                        try:
                            out.append(RelationTriple(
                                source=str(r.get("source", "")),
                                relation_type=str(r.get("relation_type", "")),
                                target=str(r.get("target", "")),
                            ))
                        except (ValueError, TypeError):
                            continue
                    return out
            return []
        except (json.JSONDecodeError, ValueError, TypeError):
            return []

    def _set_relationships(self, document: "Document", relationships: list[dict[str, Any]]) -> None:
        if document.metadata.extra is None:
            document.metadata.extra = {}
        document.metadata.extra["relationships"] = relationships

    def process(
        self,
        chunks: "list[Chunk]",
        *,
        document: "Document | None" = None,
        **kwargs: object,
    ) -> "list[Chunk]":
        """Synchronous processing not supported."""
        raise NotImplementedError(
            "EntityEntityRelationsAddOn requires async processing. Use process_async() instead."
        )

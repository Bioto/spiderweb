"""Adapter to build Entity and Relationship lists from document metadata.

Maps add-on output (e.g. document.metadata.extra["langextract"],
document.metadata.extra["relationships"]) to GraphStore input types.
Also copies source-scoping keys from document.metadata.extra into Entity.attributes
so graph queries can filter by source (e.g. source_type=x_tweet, x_user_id).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from spiderweb.models.document import Document
from spiderweb.models.graph import Entity, Relationship

# Keys from document.metadata.extra to copy onto Entity.attributes for query scoping
_SCOPE_ATTR_KEYS = frozenset(
    {"source_type", "x_tweet_id", "x_user_id", "tweet_id", "author_id", "username"}
)


def _scope_attributes_from_document(document: Document) -> dict[str, str | int | float | bool]:
    """Copy source-scoping keys from document.metadata.extra for graph/vector query filtering."""
    extra = document.metadata.extra or {}
    out: dict[str, str | int | float | bool] = {}
    for k in _SCOPE_ATTR_KEYS:
        if k not in extra:
            continue
        v = extra[k]
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif v is not None:
            out[k] = str(v)
    return out


def _normalize_attributes(raw: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Coerce attribute values to str | int | float | bool for Entity/Relationship."""
    out: dict[str, str | int | float | bool] = {}
    for k, v in raw.items():
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif v is None:
            continue
        elif isinstance(v, (list, dict)):
            out[k] = json.dumps(v, default=str)
        else:
            out[k] = str(v)
    return out


def entity_id_from_extraction(document_id: str, index: int, extraction: dict[str, Any]) -> str:
    """Compute a stable entity id for an extraction. Use this in relationship add-ons.

    Relationship add-ons should use this when building source_id/target_id so they
    match the entities we push to the graph store from LangExtract extractions.
    """
    text = extraction.get("extraction_text") or ""
    start = extraction.get("start_char")
    end = extraction.get("end_char")
    key = f"{document_id}:{index}:{text}:{start}:{end}"
    return hashlib.sha256(key.encode()).hexdigest()[:24]


def document_to_entities_and_relationships(
    document: Document,
) -> tuple[list[Entity], list[Relationship]]:
    """Extract entities and relationships from document metadata for the graph store.

    Reads:
    - document.metadata.extra["langextract"]["extractions"] -> Entity list
    - document.metadata.extra["relationships"] -> Relationship list

    Entity ids are stable (hash of document id + index + text) so relationship
    add-ons can refer to them by id if they use the same convention.

    Returns:
        (entities, relationships)
    """
    entities: list[Entity] = []
    relationships: list[Relationship] = []

    extra = document.metadata.extra or {}

    # LangExtract: extractions with extraction_class, extraction_text, attributes, start_char, end_char
    langextract = extra.get("langextract") or {}
    if isinstance(langextract, dict):
        raw_extractions = langextract.get("extractions") or []
        for i, ex in enumerate(raw_extractions):
            if not isinstance(ex, dict):
                continue
            etype = ex.get("extraction_class") or "Entity"
            text = ex.get("extraction_text") or ""
            attrs = _normalize_attributes(ex.get("attributes") or {})
            start = ex.get("start_char")
            end = ex.get("end_char")
            eid = entity_id_from_extraction(document.id, i, ex)
            scope_attrs = _scope_attributes_from_document(document)
            merged_attrs = {**attrs, **scope_attrs}
            entities.append(
                Entity(
                    id=eid,
                    type=etype,
                    label=text,
                    attributes=merged_attrs,
                    document_id=document.id,
                    chunk_id=None,
                )
            )

    # Relationships: list of {source_id, target_id, relation_type, attributes?}
    # Relationship add-on may use entity ids from LangExtract or own ids
    raw_rels = extra.get("relationships") or []
    for r in raw_rels:
        if not isinstance(r, dict):
            continue
        sid = r.get("source_id")
        tid = r.get("target_id")
        rtype = r.get("relation_type")
        if sid is None or tid is None or not rtype:
            continue
        rel_attrs = _normalize_attributes(r.get("attributes") or {})
        relationships.append(
            Relationship(
                source_id=str(sid),
                target_id=str(tid),
                relation_type=str(rtype),
                attributes=rel_attrs,
            )
        )

    return entities, relationships

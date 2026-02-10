"""Base graph store protocol.

Defines the interface for graph storage implementations (e.g. Neo4j)
that store entities and relationships produced by ingest add-ons.
"""

from typing import Protocol, runtime_checkable

from spiderweb.models.graph import Entity, Relationship


@runtime_checkable
class GraphStore(Protocol):
    """Protocol for graph storage implementations.

    Graph stores receive entities and relationships from add-ons
    (e.g. LangExtract entities, relationship extraction triples)
    during ingest.
    """

    async def upsert_entities(self, entities: list[Entity]) -> None:
        """Insert or update entities (nodes) in the graph.

        Args:
            entities: List of entities to upsert
        """
        ...

    async def upsert_relationships(self, relationships: list[Relationship]) -> None:
        """Insert or update relationships (edges) in the graph.

        Source and target entity IDs should refer to entities that
        have been (or will be) upserted via upsert_entities.

        Args:
            relationships: List of relationships to upsert
        """
        ...

    async def list_entities(
        self,
        limit: int = 100,
        type_filter: str | None = None,
    ) -> list[dict]:
        """Return entities (nodes) from the graph.

        Args:
            limit: Maximum number of entities to return.
            type_filter: Optional label/type to filter by (e.g. Person, Organization).

        Returns:
            List of dicts with at least id, type, label; may include document_id, attributes.
        """
        ...

    async def get_neighbors(
        self,
        entity_id: str,
        depth: int = 1,
    ) -> list[dict]:
        """Return relationships and adjacent nodes for an entity.

        Args:
            entity_id: Entity id to expand from.
            depth: Traversal depth (1 = direct neighbors only).

        Returns:
            List of dicts with source_id, relation_type, target_id, and optionally
            source_label, target_label for display.
        """
        ...

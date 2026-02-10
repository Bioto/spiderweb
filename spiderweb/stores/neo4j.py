"""Neo4j graph store implementation.

Stores entities and relationships from ingest add-ons (e.g. LangExtract,
relationship extraction) in Neo4j. Requires pip install spiderweb[neo4j].
"""

from __future__ import annotations

from spiderweb.models.config import GraphStoreConfig
from spiderweb.models.graph import Entity, Relationship
from spiderweb.observability.logging_config import get_logger

logger = get_logger(__name__)


class Neo4jGraphStore:
    """Neo4j-backed graph store for entities and relationships.

    Uses MERGE to upsert nodes (by id) and relationships (by source, target, type).
    Node label is derived from Entity.type; relationship type from Relationship.relation_type.
    """

    def __init__(
        self,
        uri: str,
        username: str = "neo4j",
        password: str = "",
        database: str = "neo4j",
    ):
        """Initialize Neo4j graph store.

        Args:
            uri: Neo4j connection URI (bolt:// or neo4j://)
            username: Neo4j username
            password: Neo4j password
            database: Database name
        """
        self._uri = uri
        self._auth = (username, password)
        self._database = database
        self._driver = None
        logger.debug(f"Initialized Neo4jGraphStore: uri={uri}, database={database}")

    @classmethod
    def from_config(cls, config: GraphStoreConfig) -> "Neo4jGraphStore":
        """Create store from GraphStoreConfig."""
        return cls(
            uri=config.uri,
            username=config.username,
            password=config.password,
            database=config.database,
        )

    async def _get_driver(self):
        """Lazy-init async Neo4j driver."""
        if self._driver is not None:
            return self._driver
        try:
            from neo4j import AsyncGraphDatabase

            self._driver = AsyncGraphDatabase.driver(
                self._uri,
                auth=self._auth,
            )
            logger.debug("Neo4j async driver created")
            return self._driver
        except ImportError as e:
            raise ImportError(
                "neo4j driver not installed. Install with: pip install spiderweb[neo4j]"
            ) from e

    async def upsert_entities(self, entities: list[Entity]) -> None:
        """Insert or update entities (nodes) in Neo4j.

        Uses MERGE on (n:Label {id: entity.id}) and sets properties from
        entity.label and entity.attributes. Label is derived from entity.type
        (sanitized for Cypher).
        """
        if not entities:
            return
        driver = await self._get_driver()

        def _sanitize_label(t: str) -> str:
            """Cypher labels: alphanumeric and underscore only."""
            return "".join(c if c.isalnum() or c == "_" else "_" for c in t) or "Entity"

        async with driver.session(database=self._database) as session:
            for e in entities:
                label = _sanitize_label(e.type)
                props = {"id": e.id, "label": e.label or e.id, **e.attributes}
                if e.document_id is not None:
                    props["document_id"] = e.document_id
                if e.chunk_id is not None:
                    props["chunk_id"] = e.chunk_id
                # MERGE node by id, set properties
                query = f"""
                MERGE (n:{label} {{id: $id}})
                ON CREATE SET n += $props
                ON MATCH SET n += $props
                """
                result = await session.run(query, {"id": e.id, "props": props})
                await result.consume()

        logger.debug(f"Upserted {len(entities)} entities to Neo4j")

    async def upsert_relationships(self, relationships: list[Relationship]) -> None:
        """Insert or update relationships (edges) in Neo4j.

        Matches source and target nodes by id (any label), then MERGE
        (source)-[r:REL_TYPE]->(target) and set r.attributes.
        """
        if not relationships:
            return
        driver = await self._get_driver()

        def _sanitize_rel_type(t: str) -> str:
            """Cypher relationship types: alphanumeric and underscore only."""
            return "".join(c if c.isalnum() or c == "_" else "_" for c in t) or "RELATES_TO"

        async with driver.session(database=self._database) as session:
            for rel in relationships:
                rel_type = _sanitize_rel_type(rel.relation_type)
                props = dict(rel.attributes)
                query = f"""
                MATCH (a {{id: $source_id}}), (b {{id: $target_id}})
                MERGE (a)-[r:{rel_type}]->(b)
                SET r += $props
                """
                result = await session.run(
                    query,
                    {
                        "source_id": rel.source_id,
                        "target_id": rel.target_id,
                        "props": props,
                    },
                )
                await result.consume()

        logger.debug(f"Upserted {len(relationships)} relationships to Neo4j")

    async def list_entities(
        self,
        limit: int = 100,
        type_filter: str | None = None,
    ) -> list[dict]:
        """Return entities (nodes) from Neo4j."""
        driver = await self._get_driver()
        if type_filter:
            label = "".join(
                c if c.isalnum() or c == "_" else "_" for c in type_filter
            ) or "Entity"
            query = f"""
                MATCH (n:{label})
                RETURN n.id AS id, n.label AS label, labels(n)[0] AS type,
                       n.document_id AS document_id
                LIMIT $limit
                """
        else:
            query = """
                MATCH (n)
                WHERE n.id IS NOT NULL
                RETURN n.id AS id, n.label AS label, labels(n)[0] AS type,
                       n.document_id AS document_id
                LIMIT $limit
                """
        async with driver.session(database=self._database) as session:
            result = await session.run(query, {"limit": limit})
            records = await result.data()
        return [
            {
                "id": r["id"],
                "type": r["type"] or "Entity",
                "label": r["label"] or r["id"],
                "document_id": r.get("document_id"),
            }
            for r in records
        ]

    async def get_neighbors(
        self,
        entity_id: str,
        depth: int = 1,
    ) -> list[dict]:
        """Return relationships and adjacent nodes for an entity."""
        driver = await self._get_driver()
        if depth <= 1:
            query = """
                MATCH (a {id: $entity_id})-[r]->(b)
                RETURN a.id AS source_id, a.label AS source_label,
                       type(r) AS relation_type,
                       b.id AS target_id, b.label AS target_label
                UNION
                MATCH (a)-[r]->(b {id: $entity_id})
                RETURN a.id AS source_id, a.label AS source_label,
                       type(r) AS relation_type,
                       b.id AS target_id, b.label AS target_label
                """
        else:
            query = """
                MATCH path = (start {id: $entity_id})-[*1..%d]-(end)
                WHERE start.id IS NOT NULL AND end.id IS NOT NULL
                WITH start, end, relationships(path) AS rels
                UNWIND rels AS r
                WITH startNode(r) AS src, endNode(r) AS tgt, type(r) AS rel_type
                RETURN src.id AS source_id, src.label AS source_label,
                       rel_type AS relation_type,
                       tgt.id AS target_id, tgt.label AS target_label
                LIMIT 500
                """ % min(depth, 5)
        async with driver.session(database=self._database) as session:
            result = await session.run(query, {"entity_id": entity_id})
            records = await result.data()
        return [
            {
                "source_id": r["source_id"],
                "source_label": r.get("source_label") or r["source_id"],
                "relation_type": r["relation_type"],
                "target_id": r["target_id"],
                "target_label": r.get("target_label") or r["target_id"],
            }
            for r in records
        ]

    async def close(self) -> None:
        """Close the Neo4j driver."""
        if self._driver is not None:
            await self._driver.close()
            self._driver = None
            logger.debug("Neo4j driver closed")

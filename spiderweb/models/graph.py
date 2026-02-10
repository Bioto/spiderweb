"""Graph store data models.

Entity and Relationship types for graph storage (e.g. Neo4j)
used when ingest add-ons produce entities and relationship triples.
"""

from pydantic import BaseModel, Field


class Entity(BaseModel):
    """A node in the graph (e.g. person, organization, concept)."""

    id: str = Field(description="Unique identifier for the entity")
    type: str = Field(default="Entity", description="Entity type or label (e.g. Person, Organization)")
    label: str = Field(default="", description="Display label (e.g. entity text or name)")
    attributes: dict[str, str | int | float | bool] = Field(
        default_factory=dict,
        description="Additional properties",
    )
    document_id: str | None = Field(default=None, description="Source document ID")
    chunk_id: str | None = Field(default=None, description="Source chunk ID if from a chunk")


class Relationship(BaseModel):
    """An edge between two entities."""

    source_id: str = Field(description="Source entity ID")
    target_id: str = Field(description="Target entity ID")
    relation_type: str = Field(description="Type of relationship (e.g. WORKS_AT, KNOWS)")
    attributes: dict[str, str | int | float | bool] = Field(
        default_factory=dict,
        description="Additional properties on the relationship",
    )

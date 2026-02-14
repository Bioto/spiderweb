"""Metadata queries add-on: extract structured metadata from documents for RAG filtering.

Runs a single LLM call per document using a Pydantic schema. Field names become
metadata keys; field descriptions guide the LLM. Results are stored in document.metadata.extra
and each chunk.metadata.extra so RAG search can filter (e.g. filter_dict={"doc_year": "2024"}).

Enable with ChunkAddOnConfig(enabled=["metadata_queries"], options={"metadata_queries": {...}}).
Options:
    schema: type[BaseModel] - Pydantic model class (code API). Field names = metadata keys.
    fields: list[dict] - For config/YAML: [{name, description?}]. Builds a dynamic model.
    model_id: str | None - Override LLM model for extraction.
    use_markdown_content: bool - Use document.markdown_content (default True).
    max_chars: int | None - Truncate document to this many chars before extraction (None = full doc).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, create_model

from spiderweb.config import settings
from spiderweb.observability.logging_config import get_logger

if TYPE_CHECKING:
    from spiderweb.models.document import Chunk, Document

logger = get_logger(__name__)


def _build_schema_from_fields(fields: list[dict[str, Any]]) -> type[BaseModel]:
    """Build a dynamic Pydantic model from a list of field definitions.

    Args:
        fields: List of {"name": str, "description": str | None}

    Returns:
        A Pydantic model class with optional str fields.
    """
    field_defs: dict[str, tuple[type, Any]] = {}
    for f in fields:
        name = f.get("name", "").strip()
        if not name:
            continue
        desc = f.get("description") or ""
        field_defs[name] = (str | None, Field(default=None, description=desc))
    if not field_defs:
        raise ValueError("metadata_queries fields must have at least one field with a name")
    return create_model("MetadataSchema", **field_defs)


class MetadataQueriesAddOn:
    """Add-on that extracts structured metadata from documents using a Pydantic schema.

    Runs one LLM call per document. Stores results in document.metadata.extra and
    each chunk.metadata.extra for RAG filtering.

    Example:
        >>> from pydantic import BaseModel, Field
        >>> class DocMeta(BaseModel):
        ...     doc_year: str = Field(description="What year is this document for?")
        >>> addon = MetadataQueriesAddOn(llm_client=llm, schema=DocMeta)
        >>> chunks = await addon.process_async(chunks, document=doc)
    """

    def __init__(
        self,
        llm_client: Any | None = None,
        schema: type[BaseModel] | None = None,
        schema_model: type[BaseModel] | None = None,
        fields: list[dict[str, Any]] | None = None,
        model_id: str | None = None,
        use_markdown_content: bool = True,
        max_chars: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize metadata queries add-on.

        Args:
            llm_client: GlueLLM client for extraction (required).
            schema: Pydantic model class (alias for schema_model).
            schema_model: Pydantic model class defining fields to extract (code API).
            fields: For config: list of {name, description}. Builds dynamic model.
            model_id: Override LLM model for extraction.
            use_markdown_content: Use document.markdown_content vs raw_content.
            max_chars: Max chars to send to LLM (None = full document).
            **kwargs: Ignored (for registry compatibility).
        """
        self.llm_client = llm_client
        self.model_id = model_id
        self.use_markdown_content = use_markdown_content
        self.max_chars = max_chars

        resolved_schema = schema or schema_model
        if resolved_schema is not None:
            self._schema = resolved_schema
        elif fields:
            self._schema = _build_schema_from_fields(fields)
        else:
            raise ValueError(
                "MetadataQueriesAddOn requires either schema (Pydantic model class) "
                "or fields (list of {name, description})"
            )

    async def process_async(
        self,
        chunks: "list[Chunk]",
        *,
        document: "Document | None" = None,
        **kwargs: object,
    ) -> "list[Chunk]":
        """Extract metadata from document and attach to document and all chunks."""
        if not self.llm_client:
            logger.warning(
                "MetadataQueriesAddOn: No LLM client provided, skipping metadata extraction"
            )
            return chunks

        if not document:
            logger.debug("MetadataQueriesAddOn: no document provided, skipping")
            return chunks

        text = (
            document.markdown_content if self.use_markdown_content else document.raw_content
        )
        if not text or not text.strip():
            logger.debug("MetadataQueriesAddOn: document has no content, skipping")
            return chunks

        if self.max_chars and len(text) > self.max_chars:
            text = text[: self.max_chars] + "\n\n[...truncated]"
            logger.debug(
                "MetadataQueriesAddOn: truncated document to %d chars", self.max_chars
            )

        schema = self._schema
        try:
            extracted = await self._extract_metadata(text, schema)
        except Exception as e:
            logger.error(
                "MetadataQueriesAddOn: extraction failed: %s", e, exc_info=True
            )
            document.metadata.extra["metadata_queries"] = {"error": str(e)}
            return chunks

        # model_dump() gives a dict; exclude None values so we don't pollute extra
        result = {
            k: v for k, v in extracted.model_dump().items() if v is not None
        }

        document.metadata.extra["metadata_queries"] = result

        # Merge into each chunk's extra so RAG can filter by these keys
        for chunk in chunks:
            for key, value in result.items():
                chunk.metadata.extra[key] = value

        logger.info(
            "MetadataQueriesAddOn: extracted %d fields for document %s",
            len(result),
            document.id[:8],
        )

        return chunks

    async def _extract_metadata(
        self, text: str, schema: type[BaseModel]
    ) -> BaseModel:
        """Run LLM structured extraction over document text."""
        field_descriptions = []
        for name, info in schema.model_fields.items():
            desc = getattr(info, "description", "") or ""
            field_descriptions.append(f"  - {name}: {desc}" if desc else f"  - {name}")

        prompt = f"""Extract metadata from the following document according to the schema.

Schema fields and their meanings:
{"".join(field_descriptions)}

Document:
{text}

Return a JSON object that conforms to this schema. Use short, concise values.
If a value cannot be determined from the document, use null for optional fields."""

        try:
            try:
                from gluellm.api import structured_complete

                response = await structured_complete(
                    user_message=prompt,
                    response_format=schema,
                    model=self.model_id,
                    timeout=settings.llm_timeout,
                )
                return response
            except (ImportError, AttributeError, Exception) as e:
                logger.debug(
                    "structured_complete not available, falling back to complete(): %s", e
                )
        except Exception:
            pass

        # Fallback: complete() + JSON parse
        complete_kwargs: dict[str, Any] = {
            "user_message": prompt,
            "temperature": 0.1,
            "timeout": settings.llm_timeout,
        }
        if self.model_id is not None:
            complete_kwargs["model"] = self.model_id

        try:
            response = await self.llm_client.complete(**complete_kwargs)
        except TypeError:
            complete_kwargs.pop("temperature", None)
            complete_kwargs.pop("model", None)
            complete_kwargs.pop("timeout", None)
            response = await self.llm_client.complete(**complete_kwargs)

        if hasattr(response, "final_response"):
            content = response.final_response.strip()
        elif hasattr(response, "content"):
            content = response.content.strip()
        else:
            content = str(response).strip()

        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        data = json.loads(content)
        return schema.model_validate(data)

    def process(
        self,
        chunks: "list[Chunk]",
        *,
        document: "Document | None" = None,
        **kwargs: object,
    ) -> "list[Chunk]":
        """Sync entry point; not supported. Use process_async."""
        raise NotImplementedError(
            "MetadataQueriesAddOn requires async processing. Use process_async() instead."
        )

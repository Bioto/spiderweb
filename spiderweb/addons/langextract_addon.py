"""LangExtract add-on: grounded entity extraction with source spans.

Runs LangExtract on the full document text and attaches extraction results
to document metadata (and optionally to chunks that overlap each extraction span).
Requires the optional dependency: pip install spiderweb[langextract].
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from spiderweb.observability.logging_config import get_logger

if TYPE_CHECKING:
    from spiderweb.models.document import Chunk, Document

logger = get_logger(__name__)

_LANGEXTRACT_AVAILABLE: bool | None = None


def _langextract_available() -> bool:
    """Check if langextract is installed."""
    global _LANGEXTRACT_AVAILABLE
    if _LANGEXTRACT_AVAILABLE is not None:
        return _LANGEXTRACT_AVAILABLE
    try:
        import langextract  # noqa: F401
        _LANGEXTRACT_AVAILABLE = True
    except ImportError:
        _LANGEXTRACT_AVAILABLE = False
    return _LANGEXTRACT_AVAILABLE


def _build_examples_from_options(examples_option: list[dict[str, Any]]) -> list[Any]:
    """Convert options-style examples (list of dicts) to LangExtract ExampleData list."""
    import langextract as lx
    out = []
    for ex in examples_option:
        text = ex.get("text", "")
        raw_extractions = ex.get("extractions", [])
        extractions = [
            lx.data.Extraction(
                extraction_class=e.get("extraction_class", ""),
                extraction_text=e.get("extraction_text", ""),
                attributes=e.get("attributes") or {},
            )
            for e in raw_extractions
        ]
        out.append(lx.data.ExampleData(text=text, extractions=extractions))
    return out


def _serialize_extraction(e: Any) -> dict[str, Any]:
    """Turn a single extraction (or annotated extraction) into a JSON-serializable dict."""
    if hasattr(e, "extraction_class"):
        d: dict[str, Any] = {
            "extraction_class": getattr(e, "extraction_class", ""),
            "extraction_text": getattr(e, "extraction_text", ""),
            "attributes": dict(getattr(e, "attributes", None) or {}),
        }
        if hasattr(e, "start_char") and e.start_char is not None:
            d["start_char"] = e.start_char
        if hasattr(e, "end_char") and e.end_char is not None:
            d["end_char"] = e.end_char
        return d
    if isinstance(e, dict):
        return {k: v for k, v in e.items()}
    return {"raw": str(e)}


def _extractions_from_result(result: Any) -> list[dict[str, Any]]:
    """Extract a list of serializable extraction dicts from LangExtract result."""
    out: list[dict[str, Any]] = []
    # Result may have .extractions or be an annotated document with .annotated_* or similar
    if hasattr(result, "extractions") and result.extractions:
        for e in result.extractions:
            out.append(_serialize_extraction(e))
    elif hasattr(result, "document") and hasattr(result.document, "extractions"):
        for e in result.document.extractions:
            out.append(_serialize_extraction(e))
    elif isinstance(result, list):
        for e in result:
            out.append(_serialize_extraction(e))
    return out


def _run_lx_extract_sync(
    text: str,
    prompt_description: str,
    examples: list[Any],
    model_id: str,
    api_key: str | None,
    extraction_passes: int,
    max_workers: int,
    max_char_buffer: int | None,
    **kwargs: Any,
) -> Any:
    """Run lx.extract in a sync way (to be called from asyncio.to_thread)."""
    import langextract as lx

    # Suppress verbose progress logs from the langextract library during extraction
    lx_log = logging.getLogger("langextract")
    old_level = lx_log.level
    lx_log.setLevel(logging.WARNING)
    try:
        return _run_lx_extract_impl(
            text=text,
            prompt_description=prompt_description,
            examples=examples,
            model_id=model_id,
            api_key=api_key,
            extraction_passes=extraction_passes,
            max_workers=max_workers,
            max_char_buffer=max_char_buffer,
            **kwargs,
        )
    finally:
        lx_log.setLevel(old_level)


def _run_lx_extract_impl(
    text: str,
    prompt_description: str,
    examples: list[Any],
    model_id: str,
    api_key: str | None,
    extraction_passes: int,
    max_workers: int,
    max_char_buffer: int | None,
    **kwargs: Any,
) -> Any:
    """Inner implementation of lx.extract (no logging changes)."""
    import langextract as lx
    extract_kwargs: dict[str, Any] = {
        "text_or_documents": text,
        "prompt_description": prompt_description,
        "examples": examples,
        "model_id": model_id,
        "extraction_passes": extraction_passes,
        "max_workers": max_workers,
    }
    if api_key:
        extract_kwargs["api_key"] = api_key
    if max_char_buffer is not None:
        extract_kwargs["max_char_buffer"] = max_char_buffer
    for k, v in kwargs.items():
        if v is not None and k not in extract_kwargs:
            extract_kwargs[k] = v
    return lx.extract(**extract_kwargs)


def _chunk_text(text: str, max_chars: int, overlap: int = 200) -> list[tuple[int, str]]:
    """Split text into overlapping chunks. Returns list of (start_offset, chunk_text)."""
    if max_chars <= 0 or len(text) <= max_chars:
        return [(0, text)] if text.strip() else []
    chunks: list[tuple[int, str]] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk = text[start:end]
        if chunk.strip():
            chunks.append((start, chunk))
        start = end - overlap if end < len(text) else len(text)
    return chunks


class LangExtractAddOn:
    """Add-on that runs LangExtract on document content and attaches grounded extractions.

    Runs LangExtract on the full document text (raw_content or markdown_content),
    then stores the extraction list in document.metadata.extra["langextract"].
    Optionally attaches to each chunk the extractions whose spans overlap that chunk
    (when chunk has start_char/end_char and extractions have start_char/end_char).

    Enable with ChunkAddOnConfig(enabled=["langextract"], options={"langextract": {...}}).
    Options:
        prompt_description: str (required) - What to extract (e.g. "Extract characters and emotions").
        examples: list[dict] - Few-shot examples; each dict has "text" and "extractions"
            (list of dicts with extraction_class, extraction_text, attributes).
        model_id: str - LangExtract model (default "gemini-2.5-flash"; for Ollama use e.g. "gemma2:2b").
        api_key: str | None - Override LANGEXTRACT_API_KEY (omit for local).
        model_url: str | None - For local/Ollama: base URL (e.g. "http://localhost:11434").
        fence_output: bool | None - Set False for Ollama.
        use_schema_constraints: bool | None - Set False for Ollama.
        extraction_passes: int - Number of passes for long docs (default 1).
        max_workers: int - Parallel workers (default 4).
        attach_to_chunks: bool - If True, add overlapping extractions to each chunk's
            metadata.extra["langextract_entities"] (default True).
        use_markdown_content: bool - Use document.markdown_content instead of raw_content (default True).

    Example:
        >>> config = ChunkAddOnConfig(
        ...     enabled=["langextract"],
        ...     options={
        ...         "langextract": {
        ...             "prompt_description": "Extract people, places, and dates.",
        ...             "examples": [
        ...                 {
        ...                     "text": "On Jan 1, Alice met Bob in Paris.",
        ...                     "extractions": [
        ...                         {"extraction_class": "person", "extraction_text": "Alice", "attributes": {}},
        ...                         {"extraction_class": "person", "extraction_text": "Bob", "attributes": {}},
        ...                         {"extraction_class": "place", "extraction_text": "Paris", "attributes": {}},
        ...                         {"extraction_class": "date", "extraction_text": "Jan 1", "attributes": {}},
        ...                     ],
        ...                 },
        ...             ],
        ...             "model_id": "gemini-2.5-flash",
        ...         },
        ...     },
        ... )
    """

    def __init__(
        self,
        llm_client: Any | None = None,
        prompt_description: str | None = None,
        examples: list[dict[str, Any]] | None = None,
        model_id: str = "gpt-5.1",
        api_key: str | None = None,
        extraction_passes: int = 2,
        max_workers: int = 4,
        max_char_buffer: int | None = 2000,
        attach_to_chunks: bool = True,
        use_markdown_content: bool = True,
        **kwargs: Any,
    ) -> None:
        self.llm_client = llm_client
        # Use default only when prompt_description is not provided (None); empty string means "skip"
        if prompt_description is None:
            self.prompt_description = (
                "Extract named entities: people, organizations, places, dates, and key concepts from the text."
            )
        else:
            self.prompt_description = (prompt_description or "").strip()
        self.examples_option = examples or []
        self.model_id = model_id
        self.api_key = api_key
        self.extraction_passes = extraction_passes
        self.max_workers = max_workers
        self.max_char_buffer = max_char_buffer
        self.attach_to_chunks = attach_to_chunks
        self.use_markdown_content = use_markdown_content
        self._extra_kwargs = kwargs

    async def process_async(
        self,
        chunks: "list[Chunk]",
        *,
        document: "Document | None" = None,
        **kwargs: object,
    ) -> "list[Chunk]":
        """Run LangExtract on document content and attach results to document and optionally chunks."""
        if not _langextract_available():
            logger.warning(
                "LangExtract add-on skipped: langextract not installed. "
                "Install with: pip install spiderweb[langextract]"
            )
            return chunks

        if not document:
            logger.debug("LangExtract add-on: no document provided, skipping")
            return chunks

        text = document.markdown_content if self.use_markdown_content else document.raw_content
        if not text or not text.strip():
            logger.debug("LangExtract add-on: document has no content, skipping")
            return chunks

        if not self.prompt_description.strip():
            logger.warning("LangExtract add-on: prompt_description is required; skipping")
            return chunks

        try:
            examples_option = self.examples_option
            if not examples_option:
                # LangExtract requires at least one example; use a minimal default
                examples_option = [
                    {
                        "text": "Alice works at Acme Corp. She knows Python.",
                        "extractions": [
                            {"extraction_class": "Person", "extraction_text": "Alice", "attributes": {}},
                            {"extraction_class": "Organization", "extraction_text": "Acme Corp", "attributes": {}},
                            {"extraction_class": "Skill", "extraction_text": "Python", "attributes": {}},
                        ],
                    },
                ]
                logger.debug("LangExtract add-on: using default example (none was set)")
            examples_lx = _build_examples_from_options(examples_option)
        except Exception as e:
            logger.warning("LangExtract add-on: failed to build examples: %s", e)
            return chunks

        buffer = self.max_char_buffer if self.max_char_buffer is not None else 0
        if buffer > 0 and len(text) > buffer:
            # Chunk the document ourselves so we never exceed token limits
            text_chunks = _chunk_text(text, buffer)
            all_extractions: list[dict[str, Any]] = []
            for offset, chunk_text in text_chunks:
                try:
                    result = await asyncio.to_thread(
                        _run_lx_extract_sync,
                        text=chunk_text,
                        prompt_description=self.prompt_description,
                        examples=examples_lx,
                        model_id=self.model_id,
                        api_key=self.api_key,
                        extraction_passes=1,
                        max_workers=1,
                        max_char_buffer=None,
                        **self._extra_kwargs,
                    )
                    part = _extractions_from_result(result)
                    for e in part:
                        if isinstance(e, dict):
                            if "start_char" in e and "end_char" in e:
                                e = {**e, "start_char": e["start_char"] + offset, "end_char": e["end_char"] + offset}
                            all_extractions.append(e)
                except Exception as e:
                    logger.warning("LangExtract add-on: chunk at %d failed: %s", offset, e)
            extractions = all_extractions
        else:
            try:
                result = await asyncio.to_thread(
                    _run_lx_extract_sync,
                    text=text,
                    prompt_description=self.prompt_description,
                    examples=examples_lx,
                    model_id=self.model_id,
                    api_key=self.api_key,
                    extraction_passes=self.extraction_passes,
                    max_workers=self.max_workers,
                    max_char_buffer=self.max_char_buffer,
                    **self._extra_kwargs,
                )
            except Exception as e:
                logger.error("LangExtract add-on: extraction failed: %s", e, exc_info=True)
                document.metadata.extra["langextract"] = {"error": str(e), "extractions": []}
                return chunks
            extractions = _extractions_from_result(result)
        document.metadata.extra["langextract"] = {
            "extractions": extractions,
            "model_id": self.model_id,
        }
        logger.info("LangExtract add-on: extracted %d entities for document %s", len(extractions), document.id[:8])

        if self.attach_to_chunks and extractions and chunks:
            self._attach_extractions_to_chunks(chunks, extractions)

        return chunks

    def _attach_extractions_to_chunks(
        self,
        chunks: "list[Chunk]",
        extractions: list[dict[str, Any]],
    ) -> None:
        """For each chunk, add list of extractions that overlap its span (if we have spans)."""
        for chunk in chunks:
            meta = chunk.metadata
            start = meta.start_char
            end = meta.end_char
            if start is None or end is None:
                chunk.metadata.extra["langextract_entities"] = []
                continue
            overlapping = [
                e
                for e in extractions
                if isinstance(e, dict)
                and "start_char" in e
                and "end_char" in e
                and not (e["end_char"] <= start or e["start_char"] >= end)
            ]
            chunk.metadata.extra["langextract_entities"] = overlapping

    def process(
        self,
        chunks: "list[Chunk]",
        *,
        document: "Document | None" = None,
        **kwargs: object,
    ) -> "list[Chunk]":
        """Sync entry point; not supported. Use process_async."""
        raise NotImplementedError("LangExtractAddOn requires async processing. Use process_async() instead.")

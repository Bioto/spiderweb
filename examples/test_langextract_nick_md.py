"""Standalone test: run LangExtract on _docs/nick.md and show entities + graph shape.

Run from project root:
    uv run python examples/test_langextract_nick_md.py

Requires: pip install spiderweb[langextract], OPENAI_API_KEY set.
Shows whether LangExtract produces extractions and what the graph adapter would send to Neo4j.
"""

import asyncio
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


async def main() -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    md_path = PROJECT_ROOT / "_docs" / "nick.md"
    if not md_path.exists():
        print(f"Error: {md_path} not found", file=sys.stderr)
        sys.exit(1)

    text = md_path.read_text()
    print(f"Loaded {md_path.name}: {len(text)} chars\n")

    try:
        import langextract  # noqa: F401
    except ImportError:
        print("langextract: NOT INSTALLED. pip install spiderweb[langextract]")
        sys.exit(1)

    from spiderweb.addons.langextract_addon import LangExtractAddOn
    from spiderweb.models.document import Document, DocumentMetadata
    from spiderweb.pipeline.graph_adapter import document_to_entities_and_relationships

    doc = Document(
        id="test-nick-md",
        raw_content=text,
        markdown_content=text,
        metadata=DocumentMetadata(
            source=str(md_path),
            file_type="md",
            extraction_method="file",
        ),
        chunks=[],
    )

    # Add-on with default example (no config)
    addon = LangExtractAddOn(
        prompt_description="Extract people, organizations, locations, job titles, skills, and contact info. Use extraction_class for type (e.g. Person, Organization, Skill, Place).",
        examples=[],  # add-on will inject default example
        model_id="gpt-5.1-mini",
        extraction_passes=1,
        max_workers=1,
        max_char_buffer=4000,
    )

    print("Running LangExtract add-on...")
    await addon.process_async([], document=doc)

    extra = doc.metadata.extra or {}
    lx = extra.get("langextract", {})
    err = lx.get("error")
    extractions = lx.get("extractions", [])

    if err:
        print(f"LangExtract error: {err}")
        sys.exit(1)

    print(f"Extractions: {len(extractions)}")
    if not extractions:
        print("  No entities extracted. Check prompt/model/API key.")
        sys.exit(1)

    print("\nFirst 10 extractions:")
    for i, ex in enumerate(extractions[:10], 1):
        cls_ = ex.get("extraction_class", "?")
        txt = (ex.get("extraction_text") or "?")[:50]
        print(f"  {i}. [{cls_}] {txt!r}")

    # Run graph adapter (same as pipeline) to see what would go to Neo4j
    entities, relationships = document_to_entities_and_relationships(doc)
    print(f"\nGraph adapter output (what would be sent to Neo4j):")
    print(f"  Entities: {len(entities)}")
    print(f"  Relationships: {len(relationships)} (from entity_entity_relations add-on; not run here)")
    if entities:
        print("  First 3 entities:")
        for e in entities[:3]:
            print(f"    id={e.id[:16]}... type={e.type} label={e.label!r}")

    print("\nDone. LangExtract is working.")


if __name__ == "__main__":
    asyncio.run(main())

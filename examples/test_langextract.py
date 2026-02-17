"""Test LangExtract add-on in isolation (no full ingest, no PDF).

Run from project root:
    uv run python examples/test_langextract.py

Requires: pip install spiderweb[langextract]
LangExtract uses its own API/config (e.g. LANGEXTRACT_API_KEY or model_url for local).
"""

import asyncio
import json
from pathlib import Path

# Project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Sample text that should yield clear entities
SAMPLE_TEXT = """
Nicholas Young is a senior engineer at Acme Corp in San Francisco.
He previously worked at TechStart Inc from 2019 to 2022.
Skills include Python, machine learning, and system design.
Contact: nicholas.young@example.com
"""


async def main() -> None:
    # Add project root for imports
    import sys
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from spiderweb.addons.langextract_addon import LangExtractAddOn
    from spiderweb.models.document import Document, DocumentMetadata

    # Check if langextract is available
    try:
        import langextract  # noqa: F401
        print("langextract: installed")
    except ImportError:
        print("langextract: NOT INSTALLED")
        print("  Install with: pip install spiderweb[langextract]")
        return

    # Build a minimal document
    doc = Document(
        id="test-doc-1",
        raw_content=SAMPLE_TEXT.strip(),
        markdown_content=SAMPLE_TEXT.strip(),
        metadata=DocumentMetadata(
            source="test_langextract.py",
            file_type="txt",
            extraction_method="inline",
        ),
        chunks=[],
    )

    # LangExtract requires at least one example for reliable extraction
    examples = [
        {
            "text": "Jane Doe works at Acme Corp in Boston. She knows Python and SQL.",
            "extractions": [
                {"extraction_class": "Person", "extraction_text": "Jane Doe", "attributes": {}},
                {"extraction_class": "Organization", "extraction_text": "Acme Corp", "attributes": {}},
                {"extraction_class": "Place", "extraction_text": "Boston", "attributes": {}},
                {"extraction_class": "Skill", "extraction_text": "Python", "attributes": {}},
                {"extraction_class": "Skill", "extraction_text": "SQL", "attributes": {}},
            ],
        },
    ]

    addon = LangExtractAddOn(
        prompt_description="Extract people, organizations, locations, job titles, skills, and contact info. Use extraction_class for type (e.g. Person, Organization, Skill).",
        examples=examples,
        model_id="gpt-5.1",
        extraction_passes=1,
        max_workers=1,
        max_char_buffer=0,  # use full text
    )

    print("Running LangExtract add-on on sample text...")
    chunks_out = await addon.process_async([], document=doc)
    print(f"Add-on returned {len(chunks_out)} chunks (expected 0)")

    extra = doc.metadata.extra or {}
    langextract_data = extra.get("langextract")
    if not langextract_data:
        print("\nNo document.metadata.extra['langextract'] set.")
        print("  -> Add-on likely skipped (check logs above) or failed.")
        return

    extractions = langextract_data.get("extractions", [])
    error_msg = langextract_data.get("error")

    if error_msg:
        print(f"\nLangExtract error: {error_msg}")
        print("  -> Check API key (e.g. OPENAI_API_KEY or LANGEXTRACT_API_KEY) and model_id.")
        return

    print(f"\nExtractions: {len(extractions)}")
    if not extractions:
        print("  -> No entities extracted. Try a different prompt or model.")
        return

    print("\nFirst 5 extractions:")
    for i, ex in enumerate(extractions[:5], 1):
        cls_ = ex.get("extraction_class", "?")
        text_ = ex.get("extraction_text", "?")
        print(f"  {i}. [{cls_}] {text_!r}")

    print("\nFull extractions (JSON):")
    print(json.dumps(extractions[:10], indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())

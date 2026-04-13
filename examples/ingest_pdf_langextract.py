"""Ingest a PDF with LangExtract add-on and print diagnostics.

Run from project root:
    uv run python examples/ingest_pdf_langextract.py
    uv run python examples/ingest_pdf_langextract.py _docs/10K_2024_ADBE.pdf
    uv run python examples/ingest_pdf_langextract.py _docs/10K_2024_ADBE.pdf --use-ocr

Requires: pip install spiderweb[langextract], OPENAI_API_KEY set.
For --use-ocr: pip install spiderweb[ocr]
"""

import asyncio
import sys
from pathlib import Path

# Project root (examples/ -> parent)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


async def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--use-ocr"]
    use_ocr = "--use-ocr" in sys.argv[1:]
    pdf_name = args[0] if args else "_docs/10K_2024_ADBE.pdf"
    pdf_path = (PROJECT_ROOT / pdf_name).resolve()

    if not pdf_path.exists():
        print(f"Error: PDF not found at {pdf_path}", file=sys.stderr)
        print(f"  (resolved from cwd={Path.cwd()}, arg={pdf_name})", file=sys.stderr)
        sys.exit(1)

    print(f"Ingesting: {pdf_path}")
    print(f"  Size: {pdf_path.stat().st_size / 1e6:.2f} MB")
    if use_ocr:
        print("  OCR: enabled")

    from superglue import GlueLLM
    from spiderweb import Spiderweb
    from spiderweb.models.config import ChunkAddOnConfig

    extractor = None
    if use_ocr:
        try:
            from spiderweb.extractors.ocr import OCRExtractor
            extractor = OCRExtractor(dpi=150)
        except ImportError:
            print("Error: OCR requires spiderweb[ocr]. Install with: pip install spiderweb[ocr]", file=sys.stderr)
            sys.exit(1)

    config = ChunkAddOnConfig(
        enabled=["langextract"],
        options={
            "langextract": {
                "prompt_description": "Extract company names, financial metrics, dates, and key risks. Use exact text from the document.",
                "max_char_buffer": 2000,
            },
        },
    )

    async with Spiderweb(
        llm_client=GlueLLM(),
        chunk_addon_config=config,
        extractor=extractor,
    ) as web:
        r = await web.ingest(pdf_path)

    doc = r.document
    raw_len = len(doc.raw_content)
    ex = doc.metadata.extra.get("langextract", {}).get("extractions", [])

    print()
    print("Result:")
    print(f"  Success: {r.success}")
    print(f"  Document raw_content length: {raw_len} chars")
    print(f"  Chunks created: {r.chunks_created}")
    print(f"  Chunks validated: {r.chunks_validated}")
    print(f"  LangExtract entities: {len(ex)}")
    if r.errors:
        print(f"  Errors: {r.errors}")
    if r.warnings:
        print(f"  Warnings: {r.warnings}")

    if raw_len == 0:
        print()
        print("  -> Document has no extracted text. Check PDF extraction (e.g. try --use-ocr for scanned PDFs).")
    elif r.chunks_created == 0:
        print()
        print("  -> Chunker produced 0 chunks. Check document content or chunker config.")
    elif ex:
        print()
        print("  Sample extraction:", ex[0])


if __name__ == "__main__":
    asyncio.run(main())

"""Progressive RAG processor.

Implements lazy-loading document processing where pages are summarized first
and fully processed on-demand as they're queried.
"""

import io
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gluellm import GlueLLM

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

from spiderweb.models.document import Document, DocumentMetadata
from spiderweb.models.progressive import (
    PageSummary,
    ProcessingTrigger,
    ProgressiveQueryResult,
    ProgressiveRAGConfig,
    SummaryStrategy,
)
from spiderweb.observability.logging_config import get_logger
from spiderweb.pipeline.processor import DocumentProcessor
from spiderweb.stores.base import VectorStore

logger = get_logger(__name__)


class ProgressiveRAGProcessor:
    """Process documents progressively with page summaries and on-demand full processing.

    Example:
        >>> processor = ProgressiveRAGProcessor(llm_client, summary_store, full_store)
        >>> await processor.ingest_with_summaries("document.pdf")
        >>> result = await processor.query("What is the revenue?", top_k=5)
    """

    def __init__(
        self,
        llm_client: "GlueLLM",
        summary_store: VectorStore,
        full_store: VectorStore,
        document_processor: DocumentProcessor,
        config: ProgressiveRAGConfig,
        use_ocr: bool = False,
        ocr_dpi: int = 150,
    ):
        """Initialize progressive RAG processor.

        Args:
            llm_client: GlueLLM client for embeddings and summaries
            summary_store: Vector store for page summaries
            full_store: Vector store for full chunks
            document_processor: Processor for full document chunking
            config: Progressive RAG configuration
            use_ocr: Whether to use OCR for text extraction
            ocr_dpi: DPI for OCR rendering (higher = better quality, slower)
        """
        self.llm_client = llm_client
        self.summary_store = summary_store
        self.full_store = full_store
        self.document_processor = document_processor
        self.config = config
        self.use_ocr = use_ocr
        self.ocr_dpi = ocr_dpi

        logger.info(
            f"Initialized ProgressiveRAGProcessor: strategy={config.summary_strategy.value}, "
            f"trigger={config.processing_trigger.value}, ocr={use_ocr}"
        )

    async def _extract_pages_from_pdf(self, pdf_path: Path) -> list[tuple[int, str]]:
        """Extract text from each page of a PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of (page_number, page_text) tuples
        """
        logger.info(f"Extracting pages from {pdf_path.name} (OCR={'enabled' if self.use_ocr else 'disabled'})")

        doc = fitz.open(pdf_path)
        pages = []

        for page_num in range(len(doc)):
            page = doc[page_num]

            if self.use_ocr:
                # Use OCR for text extraction
                logger.debug(f"OCR processing page {page_num + 1}/{len(doc)}")

                # Render page as image
                pix = page.get_pixmap(dpi=self.ocr_dpi)
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))

                # OCR the image
                text = pytesseract.image_to_string(img)
            else:
                # Standard text extraction
                text = page.get_text()

            pages.append((page_num + 1, text))  # 1-indexed page numbers

        doc.close()
        logger.debug(f"Extracted {len(pages)} pages from {pdf_path.name}")

        return pages

    async def _generate_summary(self, page_text: str, page_num: int) -> str:
        """Generate a summary for a page based on configured strategy.

        Args:
            page_text: Full text of the page
            page_num: Page number

        Returns:
            Summary text
        """
        if self.config.summary_strategy == SummaryStrategy.FIRST_N_CHARS:
            # Extract first N characters
            summary = page_text[: self.config.summary_length].strip()
            if len(page_text) > self.config.summary_length:
                summary += "..."
            return summary

        if self.config.summary_strategy == SummaryStrategy.LLM_SUMMARY:
            # Use LLM to generate summary
            model = self.config.llm_summary_model or "openai:gpt-4o-mini"
            prompt = f"Summarize the following page (page {page_num}) in 2-3 sentences:\n\n{page_text[:2000]}"

            response = await self.llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                max_tokens=150,
            )
            return response.choices[0].message.content.strip()

        if self.config.summary_strategy == SummaryStrategy.METADATA_ONLY:
            # Extract headings and key terms (simple implementation)
            lines = page_text.split("\n")
            # Get first few non-empty lines as "headings"
            headings = [line.strip() for line in lines[:5] if line.strip()]
            return " | ".join(headings) if headings else page_text[:200]

        return page_text[: self.config.summary_length]

    async def ingest_with_summaries(self, file_path: str | Path) -> dict:
        """Ingest a document by creating page summaries only.

        Args:
            file_path: Path to document

        Returns:
            Ingestion statistics
        """
        start_time = time.time()
        path = Path(file_path)

        logger.info(f"Starting progressive ingestion for {path.name}")

        # Extract pages
        pages = await self._extract_pages_from_pdf(path)
        document_id = f"doc-{path.stem}-{hash(path.absolute())}"

        # Generate summaries for each page
        page_summaries = []

        for page_num, page_text in pages:
            logger.debug(f"Generating summary for page {page_num}")

            # Generate summary
            summary_text = await self._generate_summary(page_text, page_num)

            # Create PageSummary object
            page_id = f"{document_id}-page-{page_num}"
            page_summary = PageSummary(
                page_id=page_id,
                document_id=document_id,
                page_number=page_num,
                summary_text=summary_text,
                metadata={
                    "source": str(path),
                    "file_type": "pdf",
                    "total_pages": len(pages),
                },
            )

            # Generate embedding for summary
            embedding_result = await self.llm_client.embed(summary_text)
            page_summary.embedding = embedding_result.embeddings[0]

            page_summaries.append(page_summary)

        # Store summaries in vector database
        # Convert PageSummary to chunks for storage
        from spiderweb.models.document import Chunk, ChunkMetadata, ChunkType

        summary_chunks = []
        for ps in page_summaries:
            chunk = Chunk(
                content=ps.summary_text,
                embedding=ps.embedding,
                metadata=ChunkMetadata(
                    document_id=ps.document_id,
                    chunk_type=ChunkType.HIERARCHICAL,
                    chunk_index=ps.page_number,
                    start_char=0,
                    end_char=len(ps.summary_text),
                    extra={
                        "page_id": ps.page_id,
                        "page_number": ps.page_number,
                        "is_summary": True,
                        "is_fully_processed": False,
                        **ps.metadata,
                    },
                ),
            )
            summary_chunks.append(chunk)

        await self.summary_store.upsert(summary_chunks)

        elapsed = time.time() - start_time
        logger.info(f"Progressive ingestion complete: {len(page_summaries)} page summaries in {elapsed:.2f}s")

        return {
            "document_id": document_id,
            "pages_summarized": len(page_summaries),
            "elapsed_time_seconds": elapsed,
            "strategy": self.config.summary_strategy.value,
        }

    async def _fully_process_page(self, page_id: str, document_id: str, page_number: int, file_path: Path) -> list:
        """Fully process a single page with the standard pipeline.

        Args:
            page_id: Page identifier
            document_id: Document identifier
            page_number: Page number to process
            file_path: Path to original document

        Returns:
            List of full chunks for the page
        """
        logger.info(f"Fully processing page {page_number} of {file_path.name}")

        # Extract just this page
        doc = fitz.open(file_path)
        page = doc[page_number - 1]  # 0-indexed

        if self.use_ocr:
            # Use OCR for text extraction
            logger.debug(f"OCR processing page {page_number} for full processing")
            pix = page.get_pixmap(dpi=self.ocr_dpi)
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            page_text = pytesseract.image_to_string(img)
        else:
            # Standard text extraction
            page_text = page.get_text()

        doc.close()

        # Create a mini-document for this page
        page_doc = Document(
            raw_content=page_text,
            markdown_content=page_text,
            metadata=DocumentMetadata(
                source=str(file_path),
                file_type="pdf",
                extraction_method="progressive_rag",
                extra={"page_number": page_number, "document_id": document_id},
            ),
        )

        # Chunk the page
        chunks = self.document_processor.chunker.chunk(page_doc)

        # Validate if enabled
        if self.document_processor.enable_validation and self.document_processor.validator:
            validation_results = await self.document_processor.validator.validate_batch(chunks)
            valid_chunks = [c for c, r in zip(chunks, validation_results, strict=True) if r.passed]
            chunks = valid_chunks

        # Generate embeddings
        if self.document_processor.enable_embedding:
            chunks = await self.document_processor._generate_embeddings(chunks)

        # Add page metadata to all chunks
        for chunk in chunks:
            chunk.metadata.extra["page_number"] = page_number
            chunk.metadata.extra["page_id"] = page_id
            chunk.metadata.extra["is_fully_processed"] = True

        # Store in full collection
        if chunks:
            await self.full_store.upsert(chunks)

        logger.info(f"Fully processed page {page_number}: {len(chunks)} chunks")

        return chunks

    async def query(self, query_text: str, top_k: int = 5, source_file: Path | None = None) -> ProgressiveQueryResult:
        """Query with progressive processing.

        First queries summaries, then optionally triggers full processing.

        Args:
            query_text: Query string
            top_k: Number of results to return
            source_file: Original document path (needed for full processing)

        Returns:
            Progressive query result with summaries and/or full chunks
        """
        start_time = time.time()

        logger.info(f"Progressive query: '{query_text[:50]}...'")

        # 1. Generate query embedding
        embedding_result = await self.llm_client.embed(query_text)
        query_embedding = embedding_result.embeddings[0]

        # 2. Query summary store
        summary_results = await self.summary_store.query(query_embedding, top_k)

        # Extract page summaries from results
        page_summaries = []
        for chunk, _score in summary_results:
            page_summary = PageSummary(
                page_id=chunk.metadata.extra.get("page_id", ""),
                document_id=chunk.metadata.document_id,
                page_number=chunk.metadata.extra.get("page_number", 0),
                summary_text=chunk.content,
                embedding=chunk.embedding,
                metadata=chunk.metadata.extra,
                is_fully_processed=chunk.metadata.extra.get("is_fully_processed", False),
            )
            page_summaries.append(page_summary)

        # 3. Check if we should trigger full processing
        newly_processed = []
        full_results = []

        if self.config.processing_trigger == ProcessingTrigger.IMMEDIATE and source_file:
            # Process pages that aren't fully processed yet
            for ps in page_summaries:
                if not ps.is_fully_processed:
                    logger.info(f"Triggering immediate processing for page {ps.page_number}")

                    # Fully process this page
                    chunks = await self._fully_process_page(ps.page_id, ps.document_id, ps.page_number, source_file)
                    full_results.extend(chunks)
                    newly_processed.append(ps.page_number)

                    # Update summary to mark as fully processed
                    # TODO: Update summary in vector store

        # 4. Query full store for already-processed pages
        if not newly_processed:
            # Check if any pages are already fully processed
            full_results_raw = await self.full_store.query(query_embedding, top_k)
            full_results = [chunk for chunk, score in full_results_raw]

        elapsed = (time.time() - start_time) * 1000

        result = ProgressiveQueryResult(
            query=query_text,
            summary_results=page_summaries,
            full_results=full_results,
            newly_processed_pages=newly_processed,
            processing_time_ms=elapsed,
            cache_hit=len(full_results) > 0 and not newly_processed,
        )

        logger.info(
            f"Progressive query complete: {len(page_summaries)} summaries, "
            f"{len(full_results)} full chunks, {len(newly_processed)} newly processed"
        )

        return result

# Chunking test fixtures

Example documents used to test the adaptive chunking agent and chunkers.

| File | Purpose | Expected strategy |
|------|---------|-------------------|
| `doc_hierarchical.md` | Markdown with clear headings | hierarchical |
| `doc_sentence.txt` | Prose / article with sentences | sentence |
| `doc_semantic.txt` | Long unstructured text, topic shifts | semantic |
| `doc_sliding_window.txt` | Simple continuous text | sliding_window |
| `doc_like_pdf.txt` | Content mimicking PDF extraction | hierarchical or sentence |
| `doc_like_ocr.txt` | Content mimicking OCR output | sentence or sliding_window |

Files use `.md` / `.txt` so tests run without binary dependencies. To test real PDFs or images, add sample files (e.g. `sample.pdf`, `sample.png`) and use the same parametrized tests; the loader will extract text via Markitdown (and OCR for images if enabled).

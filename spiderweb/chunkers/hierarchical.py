"""Hierarchical chunker for structured documents.

Splits documents based on structure (headings, sections) to preserve hierarchy.
"""

import re

from spiderweb.models.config import ChunkerConfig
from spiderweb.models.document import Chunk, ChunkMetadata, ChunkType, Document
from spiderweb.observability.logging_config import get_logger

logger = get_logger(__name__)


class HierarchicalChunker:
    """Chunker that preserves document structure.

    Splits documents based on headings and sections, creating a parent-child
    relationship between chunks at different hierarchy levels.

    Example:
        >>> chunker = HierarchicalChunker(max_chunk_size=1000)
        >>> chunks = chunker.chunk(document)
        >>> # Find top-level chunks
        >>> root_chunks = [c for c in chunks if c.parent_id is None]
    """

    def __init__(
        self,
        max_chunk_size: int = 2000,
        max_depth: int = 5,
        preserve_structure: bool = True,
    ):
        """Initialize hierarchical chunker.

        Args:
            max_chunk_size: Maximum chunk size in characters
            max_depth: Maximum hierarchy depth to preserve
            preserve_structure: Preserve section structure in chunks
        """
        self.max_chunk_size = max_chunk_size
        self.max_depth = max_depth
        self.preserve_structure = preserve_structure

        logger.debug(f"Initialized HierarchicalChunker: max_size={max_chunk_size}, max_depth={max_depth}")

    @classmethod
    def from_config(cls, config: ChunkerConfig) -> "HierarchicalChunker":
        """Create chunker from configuration.

        Args:
            config: Chunker configuration

        Returns:
            Configured chunker instance
        """
        return cls(
            max_chunk_size=config.max_chunk_size,
            max_depth=config.max_hierarchy_depth,
            preserve_structure=config.preserve_structure,
        )

    def _extract_sections(self, text: str) -> list[dict]:
        """Extract sections from markdown text.

        Args:
            text: Markdown text

        Returns:
            List of section dictionaries with title, level, content, start, end
        """
        # Match markdown headings (# Title, ## Subtitle, etc.)
        heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

        sections = []
        matches = list(heading_pattern.finditer(text))

        if not matches:
            # No headings found - treat entire document as one section
            return [
                {
                    "title": "Document",
                    "level": 0,
                    "content": text,
                    "start": 0,
                    "end": len(text),
                }
            ]

        for i, match in enumerate(matches):
            level = len(match.group(1))  # Number of # characters
            title = match.group(2).strip()
            start = match.start()

            # Content is from this heading to the next heading (or end of document)
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

            content = text[start:end].strip()

            sections.append(
                {
                    "title": title,
                    "level": level,
                    "content": content,
                    "start": start,
                    "end": end,
                }
            )

        return sections

    def _split_large_section(self, section: dict, parent_level: int) -> list[dict]:
        """Split a large section into smaller chunks.

        Args:
            section: Section dictionary
            parent_level: Level of the parent section

        Returns:
            List of subsections
        """
        content = section["content"]
        title = section["title"]

        if len(content) <= self.max_chunk_size:
            return [section]

        # Split into paragraphs first
        paragraphs = content.split("\n\n")
        subsections = []
        current_content = []
        current_length = 0

        for para in paragraphs:
            para_length = len(para)

            if current_length + para_length > self.max_chunk_size and current_content:
                # Create subsection
                subsections.append(
                    {
                        "title": f"{title} (part {len(subsections) + 1})",
                        "level": parent_level + 1,
                        "content": "\n\n".join(current_content),
                        "start": section["start"],
                        "end": section["start"] + current_length,
                    }
                )
                current_content = [para]
                current_length = para_length
            else:
                current_content.append(para)
                current_length += para_length

        # Add remaining content
        if current_content:
            subsections.append(
                {
                    "title": f"{title} (part {len(subsections) + 1})" if subsections else title,
                    "level": parent_level + 1,
                    "content": "\n\n".join(current_content),
                    "start": section["start"] + (section["end"] - current_length),
                    "end": section["end"],
                }
            )

        return subsections

    def chunk(self, document: Document) -> list[Chunk]:
        """Split document into hierarchical chunks.

        Args:
            document: Document to chunk

        Returns:
            List of chunks with parent-child relationships
        """
        text = document.markdown_content

        if not text.strip():
            logger.warning(f"Document {document.id} has no content to chunk")
            return []

        # Extract sections from markdown
        sections = self._extract_sections(text)

        # Build hierarchy
        chunks = []
        chunk_id_map = {}  # level -> last chunk id at that level

        for section in sections:
            level = min(section["level"], self.max_depth)

            # Split large sections if needed
            if len(section["content"]) > self.max_chunk_size:
                subsections = self._split_large_section(section, level)
            else:
                subsections = [section]

            for subsection in subsections:
                # Find parent (chunk at previous level)
                parent_id = None
                if level > 0:
                    for parent_level in range(level - 1, -1, -1):
                        if parent_level in chunk_id_map:
                            parent_id = chunk_id_map[parent_level]
                            break

                metadata = ChunkMetadata(
                    document_id=document.id,
                    chunk_index=len(chunks),
                    chunk_type=ChunkType.HIERARCHICAL,
                    section_title=subsection["title"],
                    section_level=level,
                    start_char=subsection["start"],
                    end_char=subsection["end"],
                )

                chunk = Chunk(
                    content=subsection["content"],
                    metadata=metadata,
                    parent_id=parent_id,
                )

                # Update parent's children list
                if parent_id:
                    for parent_chunk in chunks:
                        if parent_chunk.id == parent_id:
                            parent_chunk.children_ids.append(chunk.id)
                            break

                chunks.append(chunk)
                chunk_id_map[level] = chunk.id

        logger.info(
            f"Split document {document.id} into {len(chunks)} hierarchical chunks (max depth: {self.max_depth})"
        )

        return chunks

"""Base chunker protocol and utilities.

Defines the interface that all chunkers must implement.
"""

import re
from typing import Protocol, runtime_checkable

from spiderweb.models.document import Chunk, Document


@runtime_checkable
class Chunker(Protocol):
    """Protocol for document chunkers.

    All chunkers must implement this interface to be compatible
    with the Spiderweb processing pipeline.
    """

    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document into chunks.

        Args:
            document: Document to chunk

        Returns:
            List of chunks created from the document
        """
        ...


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences.

    Uses a simple heuristic based on punctuation and whitespace.
    For production use, consider using NLTK or spaCy for better accuracy.

    Args:
        text: Text to split

    Returns:
        List of sentences
    """
    # Simple sentence splitting on . ! ? followed by whitespace
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def split_into_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs.

    Args:
        text: Text to split

    Returns:
        List of paragraphs
    """
    paragraphs = text.split("\n\n")
    return [p.strip() for p in paragraphs if p.strip()]

"""Abstract research content store and implementations.

Allows swapping where crawled content is persisted (markdown files, Redis, DB, etc.)
so the research flow stays backend-agnostic. Only markdown is implemented for now.
"""

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from spiderweb.observability.logging_config import get_logger

logger = get_logger(__name__)


@runtime_checkable
class ResearchContentStore(Protocol):
    """Protocol for storing and retrieving research page content.

    Implementations can use markdown files, Redis, a database, etc.
    The reference returned by save_page_content is opaque to callers;
    the same store implementation must be used to retrieve content later.
    """

    def save_page_content(
        self,
        url: str,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Save page content and return a reference for later retrieval.

        Args:
            url: Page URL (used for naming or keys).
            content: Main text content (e.g. markdown).
            metadata: Optional metadata (e.g. source_query, status_code).

        Returns:
            Opaque reference (e.g. path, key, id) to pass to get_content.
        """
        ...

    def get_content(self, ref: str) -> str | None:
        """Return content for a reference from save_page_content.

        Args:
            ref: Reference returned by save_page_content.

        Returns:
            Content string, or None if missing or error.
        """
        ...


class MarkdownResearchStore:
    """Store research page content as markdown files under a base directory.

    Uses one subdirectory per "session" or trace (caller passes base_dir);
    files are named by sanitized URL + timestamp to avoid collisions.
    The reference returned is the absolute path to the .md file.
    """

    def __init__(self, base_dir: Path | str, subdir: str | None = None):
        """Initialize the markdown store.

        Args:
            base_dir: Root directory for stored files.
            subdir: Optional subdirectory under base_dir (e.g. query slug).
                    If None, files are written directly under base_dir.
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._dir = self.base_dir / subdir if subdir else self.base_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def _sanitize_filename(self, url: str) -> str:
        """Safe filename from URL with timestamp to avoid collisions."""
        import re
        from datetime import datetime

        filename = url.replace("https://", "").replace("http://", "")
        filename = re.sub(r"[^\w\-.]", "_", filename)
        if len(filename) > 200:
            filename = filename[:200]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{filename}_{timestamp}"

    def save_page_content(
        self,
        url: str,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Save content as a markdown file with optional frontmatter."""
        base_filename = self._sanitize_filename(url)
        md_file = self._dir / f"{base_filename}.md"

        if metadata:
            lines = ["---"]
            for k, v in metadata.items():
                lines.append(f"{k}: {v}")
            lines.append("---")
            lines.append("")
            body = "\n".join(lines) + content
        else:
            body = content

        md_file.write_text(body, encoding="utf-8")
        ref = str(md_file.absolute())
        logger.debug(f"Saved page content to {md_file}")
        return ref

    def get_content(self, ref: str) -> str | None:
        """Read content from a path ref; strip frontmatter if present."""
        try:
            file_content = Path(ref).read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.warning(f"Content file not found: {ref}")
            return None
        except Exception as e:
            logger.warning(f"Failed to read content from {ref}: {e}")
            return None

        if file_content.startswith("---"):
            parts = file_content.split("---", 2)
            if len(parts) >= 3:
                return parts[2].lstrip("\n")
        return file_content

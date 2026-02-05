"""Ingest cache for tracking processed files.

Tracks file hashes and modification times to enable incremental ingestion,
skipping files that haven't changed since last ingest.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from spiderweb.config import settings
from spiderweb.observability.logging_config import get_logger

logger = get_logger(__name__)


class IngestCache:
    """Cache for tracking ingested files by hash and mtime.

    Stores a mapping of file paths to their content hash, modification time,
    and document ID. Used to skip unchanged files during directory ingestion.

    Example:
        >>> cache = IngestCache()
        >>> if cache.should_process("/path/to/doc.pdf"):
        ...     # Process file
        ...     cache.update("/path/to/doc.pdf", "doc-123")
        >>> cache.save()
    """

    def __init__(self, cache_file: Path | None = None):
        """Initialize ingest cache.

        Args:
            cache_file: Path to cache file (defaults to cache_dir/ingest_cache.json)
        """
        if cache_file is None:
            cache_dir = Path(settings.cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / "ingest_cache.json"

        self.cache_file = cache_file
        self._cache: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        """Load cache from disk."""
        if not self.cache_file.exists():
            logger.debug(f"Cache file {self.cache_file} does not exist, starting fresh")
            return

        try:
            with open(self.cache_file, encoding="utf-8") as f:
                self._cache = json.load(f)
            logger.debug(f"Loaded {len(self._cache)} entries from cache")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}, starting fresh")
            self._cache = {}

    def save(self) -> None:
        """Save cache to disk."""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2)
            logger.debug(f"Saved {len(self._cache)} entries to cache")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}", exc_info=True)

    def _get_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file contents.

        Args:
            file_path: Path to file

        Returns:
            Hex digest of file hash
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _get_file_mtime(self, file_path: Path) -> float:
        """Get file modification time.

        Args:
            file_path: Path to file

        Returns:
            Modification time as float
        """
        return file_path.stat().st_mtime

    def should_process(self, file_path: Path, force: bool = False) -> bool:
        """Check if file should be processed.

        Args:
            file_path: Path to file
            force: If True, always return True (bypass cache)

        Returns:
            True if file should be processed, False if unchanged
        """
        if force:
            return True

        if not file_path.exists():
            return False

        file_str = str(file_path.resolve())
        current_hash = self._get_file_hash(file_path)
        current_mtime = self._get_file_mtime(file_path)

        # Check cache
        if file_str in self._cache:
            cached = self._cache[file_str]
            if (
                cached.get("hash") == current_hash
                and cached.get("mtime") == current_mtime
            ):
                logger.debug(f"Skipping unchanged file: {file_path}")
                return False

        return True

    def update(self, file_path: Path, document_id: str) -> None:
        """Update cache entry for a processed file.

        Args:
            file_path: Path to file
            document_id: Document ID from ingestion
        """
        if not file_path.exists():
            logger.warning(f"Cannot cache non-existent file: {file_path}")
            return

        file_str = str(file_path.resolve())
        self._cache[file_str] = {
            "hash": self._get_file_hash(file_path),
            "mtime": self._get_file_mtime(file_path),
            "document_id": document_id,
        }

    def get_document_id(self, file_path: Path) -> str | None:
        """Get cached document ID for a file.

        Args:
            file_path: Path to file

        Returns:
            Document ID if cached, None otherwise
        """
        file_str = str(file_path.resolve())
        entry = self._cache.get(file_str)
        return entry.get("document_id") if entry else None

    def remove(self, file_path: Path) -> None:
        """Remove cache entry for a file.

        Args:
            file_path: Path to file
        """
        file_str = str(file_path.resolve())
        if file_str in self._cache:
            del self._cache[file_str]
            logger.debug(f"Removed cache entry for: {file_path}")

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        logger.info("Cleared ingest cache")

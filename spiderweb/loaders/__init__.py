"""Document loaders for Spiderweb.

This package provides loaders for reading documents from various sources.
"""

from spiderweb.loaders.directory_loader import DirectoryLoader
from spiderweb.loaders.file_loader import FileLoader

__all__ = ["FileLoader", "DirectoryLoader"]

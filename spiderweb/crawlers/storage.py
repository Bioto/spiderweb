"""File-based storage for crawled web content.

Provides utilities for saving crawled content to local filesystem
in various formats (JSON, markdown, HTML, etc.).
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from spiderweb.crawlers.base import CrawlResult
from spiderweb.models.document import Document
from spiderweb.observability.logging_config import get_logger

logger = get_logger(__name__)


class CrawlStorage:
    """Storage handler for crawled web content.
    
    Saves crawled data to local filesystem in various formats.
    
    Example:
        >>> storage = CrawlStorage(output_dir="./crawled_data")
        >>> await storage.save_crawl_result(result, format="markdown")
    """
    
    def __init__(self, output_dir: str | Path):
        """Initialize crawl storage.
        
        Args:
            output_dir: Directory to store crawled content
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized CrawlStorage at {self.output_dir}")
    
    def _sanitize_filename(self, url: str) -> str:
        """Convert URL to safe filename.
        
        Args:
            url: URL to sanitize
            
        Returns:
            Safe filename string
        """
        # Remove protocol
        filename = url.replace("https://", "").replace("http://", "")
        
        # Replace special characters with underscores
        filename = re.sub(r'[^\w\-.]', '_', filename)
        
        # Limit length
        if len(filename) > 200:
            filename = filename[:200]
        
        # Add timestamp to avoid collisions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return f"{filename}_{timestamp}"
    
    def save_crawl_result(
        self,
        result: CrawlResult,
        format: str = "all",
        include_metadata: bool = True,
    ) -> dict[str, Path]:
        """Save crawl result to disk.
        
        Args:
            result: CrawlResult to save
            format: Output format - "markdown", "html", "json", or "all"
            include_metadata: Include metadata in output
            
        Returns:
            Dictionary of format -> saved file path
        """
        base_filename = self._sanitize_filename(result.url)
        saved_files = {}
        
        # Save markdown
        if format in ("markdown", "all") and result.markdown:
            md_file = self.output_dir / f"{base_filename}.md"
            content = result.markdown
            
            if include_metadata:
                # Add frontmatter
                metadata_lines = [
                    "---",
                    f"url: {result.url}",
                    f"crawled_at: {datetime.now().isoformat()}",
                    f"status_code: {result.status_code}",
                    f"success: {result.success}",
                ]
                if result.links:
                    metadata_lines.append(f"links_found: {len(result.links)}")
                metadata_lines.append("---")
                metadata_lines.append("")
                
                content = "\n".join(metadata_lines) + content
            
            md_file.write_text(content, encoding="utf-8")
            saved_files["markdown"] = md_file
            logger.debug(f"Saved markdown to {md_file}")
        
        # Save HTML
        if format in ("html", "all") and result.content:
            html_file = self.output_dir / f"{base_filename}.html"
            html_file.write_text(result.content, encoding="utf-8")
            saved_files["html"] = html_file
            logger.debug(f"Saved HTML to {html_file}")
        
        # Save JSON (complete data)
        if format in ("json", "all"):
            json_file = self.output_dir / f"{base_filename}.json"
            
            data = {
                "url": result.url,
                "crawled_at": datetime.now().isoformat(),
                "status_code": result.status_code,
                "success": result.success,
                "error": result.error,
                "content": result.content if format == "all" else None,
                "markdown": result.markdown,
                "metadata": result.metadata,
                "links": result.links[:100] if result.links else [],  # Limit links
            }
            
            json_file.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
            saved_files["json"] = json_file
            logger.debug(f"Saved JSON to {json_file}")
        
        logger.info(f"Saved {len(saved_files)} file(s) for {result.url}")
        return saved_files
    
    def save_document(
        self,
        document: Document,
        format: str = "all",
        include_chunks: bool = False,
    ) -> dict[str, Path]:
        """Save processed document to disk.
        
        Args:
            document: Document to save
            format: Output format
            include_chunks: Include chunk data in output
            
        Returns:
            Dictionary of format -> saved file path
        """
        base_filename = self._sanitize_filename(document.metadata.source)
        saved_files = {}
        
        # Save markdown
        if format in ("markdown", "all"):
            md_file = self.output_dir / f"{base_filename}.md"
            content = document.markdown_content
            
            # Add frontmatter
            metadata_lines = [
                "---",
                f"source: {document.metadata.source}",
                f"file_type: {document.metadata.file_type}",
                f"extraction_method: {document.metadata.extraction_method}",
                f"processed_at: {document.metadata.created_at.isoformat()}",
                f"chunks: {len(document.chunks)}",
                "---",
                "",
            ]
            
            content = "\n".join(metadata_lines) + content
            md_file.write_text(content, encoding="utf-8")
            saved_files["markdown"] = md_file
        
        # Save JSON (complete data)
        if format in ("json", "all"):
            json_file = self.output_dir / f"{base_filename}.json"
            
            data = {
                "id": document.id,
                "source": document.metadata.source,
                "metadata": document.metadata.model_dump(),
                "raw_content": document.raw_content if format == "all" else None,
                "markdown_content": document.markdown_content,
            }
            
            if include_chunks:
                data["chunks"] = [
                    {
                        "id": chunk.id,
                        "content": chunk.content,
                        "metadata": chunk.metadata.model_dump(),
                        "has_embedding": chunk.embedding is not None,
                        "validation_scores": chunk.validation_scores,
                    }
                    for chunk in document.chunks
                ]
            
            json_file.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
            saved_files["json"] = json_file
        
        logger.info(f"Saved {len(saved_files)} file(s) for document {document.id}")
        return saved_files
    
    def save_structured_data(
        self,
        data: BaseModel | dict,
        url: str,
        format: str = "json",
    ) -> Path:
        """Save structured extracted data.
        
        Args:
            data: Pydantic model or dict to save
            url: Source URL
            format: Output format ("json" or "yaml")
            
        Returns:
            Path to saved file
        """
        base_filename = self._sanitize_filename(url)
        
        if format == "json":
            file_path = self.output_dir / f"{base_filename}_data.json"
            
            if isinstance(data, BaseModel):
                json_data = data.model_dump()
            else:
                json_data = data
            
            file_path.write_text(
                json.dumps(json_data, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
        
        elif format == "yaml":
            try:
                import yaml
                file_path = self.output_dir / f"{base_filename}_data.yaml"
                
                if isinstance(data, BaseModel):
                    yaml_data = data.model_dump()
                else:
                    yaml_data = data
                
                file_path.write_text(
                    yaml.dump(yaml_data, allow_unicode=True, default_flow_style=False),
                    encoding="utf-8"
                )
            except ImportError:
                logger.warning("PyYAML not installed, falling back to JSON")
                return self.save_structured_data(data, url, format="json")
        
        logger.info(f"Saved structured data to {file_path}")
        return file_path
    
    def create_index(self) -> Path:
        """Create an index file listing all saved content.
        
        Returns:
            Path to index file
        """
        index_file = self.output_dir / "index.json"
        
        index_data = {
            "created_at": datetime.now().isoformat(),
            "output_dir": str(self.output_dir),
            "files": []
        }
        
        # Scan directory
        for file_path in self.output_dir.glob("*"):
            if file_path.name == "index.json":
                continue
            
            if file_path.is_file():
                index_data["files"].append({
                    "name": file_path.name,
                    "size": file_path.stat().st_size,
                    "modified": datetime.fromtimestamp(
                        file_path.stat().st_mtime
                    ).isoformat(),
                })
        
        index_file.write_text(
            json.dumps(index_data, indent=2),
            encoding="utf-8"
        )
        
        logger.info(f"Created index with {len(index_data['files'])} files")
        return index_file



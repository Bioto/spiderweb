"""Example: Web Crawling with Local File Storage

This example demonstrates how to:
1. Crawl web content with crawl4ai
2. Save crawled content to local files (markdown, HTML, JSON)
3. Optionally ingest to vector store as well
4. Access and process saved files

Requirements:
    - Set OPENAI_API_KEY in environment
    - Run: playwright install chromium (for crawl4ai)
"""

import asyncio
import json
from pathlib import Path

from gluellm import GlueLLM

from spiderweb import Spiderweb
from spiderweb.crawlers.storage import CrawlStorage
from spiderweb.models.config import CrawlerConfig


async def main():
    """Main execution function."""
    # Initialize LLM client
    llm = GlueLLM()
    web = Spiderweb(llm_client=llm)
    
    # ====================================================================
    # Example 1: Basic crawl with local storage (markdown only)
    # ====================================================================
    print("\n" + "="*70)
    print("Example 1: Save crawled content as markdown")
    print("="*70 + "\n")
    
    result = await web.crawl(
        url="https://example.com",
        crawler_config=CrawlerConfig(provider="crawl4ai"),
        save_to="./crawled_data/example1",
        save_format="markdown"
    )
    
    print(f"✓ Crawled and saved to markdown")
    print(f"  URL: {result.url}")
    print(f"  Status: {result.status_code}")
    print(f"  Saved at: ./crawled_data/example1/")
    
    # ====================================================================
    # Example 2: Save all formats (markdown, HTML, JSON)
    # ====================================================================
    print("\n" + "="*70)
    print("Example 2: Save in all formats")
    print("="*70 + "\n")
    
    result = await web.crawl(
        url="https://example.com",
        crawler_config=CrawlerConfig(provider="crawl4ai"),
        save_to="./crawled_data/example2",
        save_format="all"  # Saves .md, .html, and .json
    )
    
    print(f"✓ Saved in all formats")
    saved_dir = Path("./crawled_data/example2")
    for file in saved_dir.glob("*"):
        print(f"  {file.name} ({file.stat().st_size} bytes)")
    
    # ====================================================================
    # Example 3: Crawl multiple pages with depth
    # ====================================================================
    print("\n" + "="*70)
    print("Example 3: Crawl multiple pages with link following")
    print("="*70 + "\n")
    
    results = await web.crawl(
        url="https://example.com",
        crawler_config=CrawlerConfig(
            provider="crawl4ai",
            max_depth=2,  # Follow links 2 levels deep
            max_pages=5,  # Stop after 5 pages
        ),
        save_to="./crawled_data/example3",
        save_format="json"
    )
    
    print(f"✓ Crawled {len(results)} pages")
    saved_dir = Path("./crawled_data/example3")
    for file in saved_dir.glob("*.json"):
        data = json.loads(file.read_text())
        print(f"  {data['url']} - {data['status_code']}")
    
    # ====================================================================
    # Example 4: Save locally AND ingest to vector store
    # ====================================================================
    print("\n" + "="*70)
    print("Example 4: Hybrid storage (local files + vector store)")
    print("="*70 + "\n")
    
    # This saves to local files AND ingests into Qdrant
    result = await web.crawl(
        url="https://example.com",
        crawler_config=CrawlerConfig(provider="crawl4ai"),
        save_to="./crawled_data/example4",
        save_format="all",
        ingest=True,  # Also ingest to vector store
    )
    
    print(f"✓ Saved locally AND ingested to vector store")
    print(f"  Documents processed: {result.successful_documents}")
    print(f"  Chunks created: {result.total_chunks}")
    print(f"  Local files saved: ./crawled_data/example4/")
    
    # ====================================================================
    # Example 5: Using CrawlStorage directly for custom processing
    # ====================================================================
    print("\n" + "="*70)
    print("Example 5: Direct use of CrawlStorage")
    print("="*70 + "\n")
    
    from spiderweb.crawlers.http import HttpCrawler
    
    # Create crawler and storage
    crawler = HttpCrawler()
    storage = CrawlStorage(output_dir=Path("./crawled_data/example5"))
    
    # Crawl
    crawl_result = await crawler.crawl(
        "https://example.com",
        CrawlerConfig(provider="http")
    )
    
    # Save with custom format
    saved_files = storage.save_crawl_result(
        crawl_result,
        format="markdown",
        include_metadata=True
    )
    
    print(f"✓ Custom save with CrawlStorage")
    for fmt, path in saved_files.items():
        print(f"  {fmt}: {path}")
    
    # Create index of all saved files
    index_file = storage.create_index()
    print(f"  Index created: {index_file}")
    
    # ====================================================================
    # Example 6: Processing saved files
    # ====================================================================
    print("\n" + "="*70)
    print("Example 6: Process saved JSON files")
    print("="*70 + "\n")
    
    crawled_dir = Path("./crawled_data")
    
    # Find all JSON files
    json_files = list(crawled_dir.rglob("*.json"))
    
    print(f"Found {len(json_files)} JSON files across all examples")
    
    # Process each file
    total_links = 0
    for json_file in json_files:
        if json_file.name == "index.json":
            continue  # Skip index files
            
        try:
            data = json.loads(json_file.read_text())
            total_links += len(data.get("links", []))
        except Exception as e:
            print(f"  Error reading {json_file}: {e}")
    
    print(f"Total links discovered across all crawls: {total_links}")
    
    # ====================================================================
    # Example 7: Using with Git for version control
    # ====================================================================
    print("\n" + "="*70)
    print("Example 7: Git-based documentation archiving")
    print("="*70 + "\n")
    
    # Crawl documentation to git-tracked directory
    docs_dir = Path("./docs_archive")
    docs_dir.mkdir(exist_ok=True)
    
    result = await web.crawl(
        url="https://example.com",
        crawler_config=CrawlerConfig(provider="crawl4ai"),
        save_to=str(docs_dir),
        save_format="markdown"
    )
    
    print(f"✓ Saved documentation for git tracking")
    print(f"  Directory: {docs_dir}")
    print(f"  Files: {len(list(docs_dir.glob('*.md')))}")
    print("\n  To version control these:")
    print(f"    cd {docs_dir}")
    print("    git add .")
    print("    git commit -m 'Crawled docs on $(date)'")
    print("    git push")
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)
    
    # Cleanup (optional)
    # import shutil
    # shutil.rmtree("./crawled_data", ignore_errors=True)
    # shutil.rmtree("./docs_archive", ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())




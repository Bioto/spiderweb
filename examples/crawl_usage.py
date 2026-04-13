"""Example usage of Spiderweb's web crawling features.

Demonstrates:
- Basic web crawling
- Structured extraction with Pydantic schemas
- Crawl and query (RAG with fresh web content)
- Link following and depth control
"""

import asyncio
from pydantic import BaseModel, Field
from superglue import GlueLLM
from spiderweb import Spiderweb
from spiderweb.models.config import (
    CrawlerConfig,
    CrawlExtractionConfig,
)


# Define a Pydantic schema for structured extraction
class Article(BaseModel):
    """Schema for extracting article information from web pages."""
    title: str = Field(description="Article title or heading")
    author: str | None = Field(default=None, description="Article author if available")
    summary: str = Field(description="Brief summary of the article content")
    main_topics: list[str] = Field(default_factory=list, description="Main topics or themes")
    publication_date: str | None = Field(default=None, description="Publication date if available")


class ProductInfo(BaseModel):
    """Schema for extracting product information from e-commerce pages."""
    name: str = Field(description="Product name")
    price: float = Field(description="Product price in dollars")
    description: str = Field(description="Product description")
    features: list[str] = Field(default_factory=list, description="Key product features")
    availability: str | None = Field(default=None, description="Stock availability status")


async def basic_crawl_example():
    """Example 1: Basic web crawling."""
    print("\n=== Example 1: Basic Web Crawling ===\n")
    
    async with Spiderweb(llm_client=GlueLLM()) as web:
        # Simple single-page crawl
        result = await web.crawl("https://example.com")
        
        print(f"Crawled: {result.url}")
        print(f"Status: {result.status_code}")
        print(f"Content length: {len(result.content)} bytes")
        print(f"Links found: {len(result.links)}")
        
        if result.markdown:
            print("\nMarkdown preview:")
            print(result.markdown[:300] + "...")


async def structured_extraction_example():
    """Example 2: Structured extraction with Pydantic schema."""
    print("\n=== Example 2: Structured Extraction ===\n")
    
    async with Spiderweb(llm_client=GlueLLM()) as web:
        # Crawl and extract structured data
        result = await web.crawl(
            "https://example.com/article",
            output_schema=Article,
            extraction_config=CrawlExtractionConfig(
                semantic_guide="Extract article metadata and content summary",
                auto_improve=True,  # Enables iterative improvement
                max_improve_iterations=3,
            ),
        )
        
        print("Extracted article data:")
        print(f"Title: {result.extracted_data.title}")
        print(f"Summary: {result.extracted_data.summary}")
        print(f"Topics: {', '.join(result.extracted_data.main_topics)}")


async def crawl_with_depth_example():
    """Example 3: Crawling with link following."""
    print("\n=== Example 3: Link Following ===\n")
    
    async with Spiderweb(llm_client=GlueLLM()) as web:
        # Crawl multiple pages by following links
        results = await web.crawl(
            "https://example.com/docs",
            crawler_config=CrawlerConfig(
                max_depth=2,  # Follow links 2 levels deep
                max_pages=20,  # Limit total pages
                follow_patterns=[r"example\.com/docs"],  # Only follow docs links
                exclude_patterns=[r"/archive/", r"/old/"],  # Exclude certain paths
                delay_between_requests=1.0,  # Be polite
            ),
        )
        
        print(f"Crawled {len(results)} pages:")
        for result in results[:5]:  # Show first 5
            print(f"  - {result.url} ({result.status_code})")


async def crawl_and_ingest_example():
    """Example 4: Crawl and ingest into vector store."""
    print("\n=== Example 4: Crawl and Ingest ===\n")
    
    async with Spiderweb(
        llm_client=GlueLLM(),
        vector_store_url="qdrant://localhost:6333/web_docs",
    ) as web:
        # Crawl and automatically ingest into vector store
        result = await web.crawl(
            ["https://example.com/docs/intro", "https://example.com/docs/guide"],
            crawler_config=CrawlerConfig(max_depth=2),
            ingest=True,  # Automatically chunk, embed, and store
        )
        
        print(f"Ingested {result.total_documents} documents")
        print(f"Created {result.total_chunks} chunks")
        print(f"Processing time: {result.processing_time_seconds:.2f}s")


async def crawl_and_query_example():
    """Example 5: Crawl fresh content and query with existing data."""
    print("\n=== Example 5: Crawl and Query (RAG with Fresh Data) ===\n")
    
    async with Spiderweb(
        llm_client=GlueLLM(),
        vector_store_url="qdrant://localhost:6333/knowledge_base",
    ) as web:
        # First, let's say we have existing documents ingested...
        # await web.ingest_directory("./docs")
        
        # Now crawl fresh web content and query it along with existing data
        results = await web.crawl_and_query(
            query="What are the latest pricing updates?",
            urls=[
                "https://company.com/pricing",
                "https://company.com/blog/latest",
            ],
            crawler_config=CrawlerConfig(max_depth=1),
            extraction_config=CrawlExtractionConfig(
                semantic_guide="Focus on pricing information and recent changes",
            ),
            top_k=10,
        )
        
        print(f"Found {len(results.chunks)} relevant chunks:")
        for i, chunk in enumerate(results.chunks[:3], 1):
            print(f"\n{i}. Score: {results.scores[i-1]:.3f}")
            print(f"   Source: {chunk['metadata'].get('source', 'unknown')}")
            print(f"   Content: {chunk['content'][:200]}...")


async def product_extraction_example():
    """Example 6: E-commerce product extraction."""
    print("\n=== Example 6: Product Information Extraction ===\n")
    
    async with Spiderweb(llm_client=GlueLLM()) as web:
        result = await web.crawl(
            "https://store.example.com/product/widget-pro",
            output_schema=ProductInfo,
            extraction_config=CrawlExtractionConfig(
                semantic_guide="Extract complete product information including pricing and features",
                extraction_query="Focus on finding the exact price, all listed features, and stock status",
                auto_improve=True,
            ),
        )
        
        print("Extracted product information:")
        print(f"Name: {result.extracted_data.name}")
        print(f"Price: ${result.extracted_data.price:.2f}")
        print(f"Description: {result.extracted_data.description[:100]}...")
        print(f"Features: {len(result.extracted_data.features)} features found")
        print(f"Availability: {result.extracted_data.availability or 'Unknown'}")


async def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("Spiderweb Web Crawling Examples")
    print("="*60)
    
    try:
        # Run examples (comment out as needed for testing)
        await basic_crawl_example()
        # await structured_extraction_example()
        # await crawl_with_depth_example()
        # await crawl_and_ingest_example()
        # await crawl_and_query_example()
        # await product_extraction_example()
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)


if __name__ == "__main__":
    asyncio.run(main())


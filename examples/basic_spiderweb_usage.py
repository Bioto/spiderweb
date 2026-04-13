"""Basic usage example for Spiderweb.

This example demonstrates the basic workflow of ingesting documents
and querying them using Spiderweb.
"""

import asyncio

from superglue import GlueLLM

from spiderweb import Spiderweb


async def basic_example():
    """Basic document ingestion and querying."""
    print("=== Basic Spiderweb Example ===\n")

    # Initialize LLM client for embeddings
    llm = GlueLLM()

    # Example 1: Quick single file ingestion
    print("1. Single file ingestion")
    # doc = await ingest("document.pdf", llm_client=llm)
    # print(f"   Created {len(doc.chunks)} chunks\n")

    # Example 2: Using Spiderweb client
    print("2. Using Spiderweb client")
    async with Spiderweb(llm_client=llm):
        # Ingest a document
        # result = await web.ingest("document.pdf")
        # print(f"   Chunks created: {result.chunks_created}")
        # print(f"   Chunks validated: {result.chunks_validated}")
        # print(f"   Processing time: {result.processing_time_seconds:.2f}s\n")

        # Query the store
        # results = await web.query("What is the main topic?", top_k=3)
        # print(f"   Found {results.total_results} results:")
        # for chunk in results.chunks[:2]:
        #     print(f"   - {chunk['content'][:100]}...")
        pass

    print("\n✓ Example complete!")


async def batch_example():
    """Batch processing example."""
    print("\n=== Batch Processing Example ===\n")

    # Process an entire directory
    print("Processing directory...")
    # result = await process_directory(
    #     "/path/to/docs",
    #     llm_client=GlueLLM(),
    #     chunker="semantic",
    #     recursive=True,
    # )
    # print(f"Processed {result.successful_documents}/{result.total_documents} documents")
    # print(f"Total chunks: {result.total_chunks}")
    # print(f"Average time per document: {result.average_time_per_document:.2f}s")

    print("\n✓ Batch example complete!")


async def qdrant_example():
    """Example using Qdrant vector store."""
    print("\n=== Qdrant Vector Store Example ===\n")

    # Create Spiderweb with Qdrant
    # Uncomment to use:
    # web = Spiderweb(
    #     llm_client=GlueLLM(),
    #     vector_store_url="qdrant://localhost:6333/my_documents",
    # )

    # Ingest documents
    print("Ingesting documents to Qdrant...")
    # result = await web.ingest("document.pdf")
    # print(f"Stored {result.chunks_validated} chunks in Qdrant\n")

    # Query
    print("Querying Qdrant...")
    # results = await web.query("machine learning best practices", top_k=5)
    # for idx, chunk in enumerate(results.chunks, 1):
    #     print(f"{idx}. {chunk['content'][:80]}...")

    print("\n✓ Qdrant example complete!")


async def advanced_example():
    """Advanced configuration example."""
    print("\n=== Advanced Configuration Example ===\n")

    from spiderweb.models.config import ChunkerConfig, ValidatorConfig

    # Configure chunking (example configuration)
    _chunker_config = ChunkerConfig(
        strategy="hierarchical",
        max_chunk_size=1500,
        chunk_overlap=300,
        preserve_structure=True,
    )

    # Configure validation (example configuration)
    _validator_config = ValidatorConfig(
        enable_validation=True,
        min_quality_score=0.5,
        enable_deduplication=True,
        deduplication_threshold=0.95,
    )

    # Create configured client (uncomment to use)
    # web = Spiderweb(
    #     llm_client=GlueLLM(),
    #     chunker_config=chunker_config,
    #     validator_config=validator_config,
    # )

    print("Configured with:")
    print("  - Hierarchical chunking (max size: 1500)")
    print("  - Quality validation (min score: 0.5)")
    print("  - Deduplication (threshold: 0.95)")

    # Process document with custom configuration
    # result = await web.ingest("document.pdf")
    # print(f"\nProcessed document:")
    # print(f"  Chunks created: {result.chunks_created}")
    # print(f"  Chunks validated: {result.chunks_validated}")
    # print(f"  Chunks rejected: {result.chunks_rejected}")

    print("\n✓ Advanced example complete!")


async def main():
    """Run all examples."""
    await basic_example()
    # await batch_example()
    # await qdrant_example()
    # await advanced_example()


if __name__ == "__main__":
    asyncio.run(main())

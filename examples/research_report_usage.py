"""Example usage of Spiderweb's research agent features.

Demonstrates:
- Persona-driven research with report synthesis
- Goal-driven research with planning
- Structured report output with Pydantic schemas
- Parallel crawler execution
"""

import asyncio
from pydantic import BaseModel, Field
from superglue import GlueLLM
from spiderweb import Spiderweb
from spiderweb.models.config import (
    ResearchAgentConfig,
    SearchProviderConfig,
    CrawlerConfig,
    SearchDepthConfig,
)


# Define a Pydantic schema for structured report output
class CompetitorAnalysis(BaseModel):
    """Structured competitor analysis report."""
    company_name: str = Field(description="Name of the competitor")
    pricing_tiers: list[dict] = Field(description="List of pricing tiers with details")
    key_features: list[str] = Field(description="Key product features")
    market_positioning: str = Field(description="How they position themselves in the market")
    strengths: list[str] = Field(description="Competitive strengths")
    weaknesses: list[str] = Field(description="Competitive weaknesses")


class MarketResearchReport(BaseModel):
    """Structured market research report."""
    topic: str = Field(description="Research topic")
    key_findings: list[str] = Field(description="Main findings from research")
    trends: list[str] = Field(description="Identified trends")
    recommendations: list[str] = Field(description="Recommendations based on findings")


async def persona_driven_research_example():
    """Example 1: Persona-driven research with free-form report."""
    print("\n=== Example 1: Persona-Driven Research ===\n")
    
    async with Spiderweb(llm_client=GlueLLM()) as web:
        result = await web.research_and_report(
            persona="You are a market research analyst specializing in SaaS",
            instructions="Research competitor pricing strategies in the SaaS industry, focusing on freemium models and enterprise tiers",
            research_config=ResearchAgentConfig(
                num_queries=5,
                max_parallel_crawls=3,
            ),
        )
        
        print(f"Generated {len(result.queries_used)} queries:")
        for i, query in enumerate(result.queries_used, 1):
            print(f"  {i}. {query}")
        
        print(f"\nCrawled {sum(len(t.get_all_urls()) for t in result.traces)} URLs total")
        print(f"\nReport ({len(result.report)} chars):")
        print("=" * 80)
        print(result.report[:1000] + "..." if len(result.report) > 1000 else result.report)
        print("=" * 80)


async def goal_driven_research_example():
    """Example 2: Goal-driven research with planning."""
    print("\n=== Example 2: Goal-Driven Research ===\n")
    
    async with Spiderweb(llm_client=GlueLLM()) as web:
        result = await web.achieve_goal(
            goal="Understand competitor X's pricing and positioning in the EU market",
            persona="You are a strategic analyst",
            research_config=ResearchAgentConfig(
                num_queries=4,
                max_parallel_crawls=3,
            ),
        )
        
        print("Research Plan:")
        print(f"  Queries: {result.plan.queries}")
        print(f"  Report Focus: {result.plan.report_focus}")
        if result.plan.rationale:
            print(f"  Rationale: {result.plan.rationale}")
        
        print(f"\nExecuted {len(result.queries_used)} queries")
        print(f"Crawled {sum(len(t.get_all_urls()) for t in result.traces)} URLs")
        print(f"\nReport ({len(result.report)} chars):")
        print("=" * 80)
        print(result.report[:1000] + "..." if len(result.report) > 1000 else result.report)
        print("=" * 80)


async def structured_report_example():
    """Example 3: Structured report output with Pydantic schema."""
    print("\n=== Example 3: Structured Report Output ===\n")
    
    async with Spiderweb(llm_client=GlueLLM()) as web:
        result = await web.research_and_report(
            persona="You are a competitive intelligence analyst",
            instructions="Research a major SaaS competitor's product features, pricing, and market positioning",
            research_config=ResearchAgentConfig(num_queries=5),
            report_schema=CompetitorAnalysis,
        )
        
        # Access structured data
        analysis = result.report  # This is a CompetitorAnalysis instance
        
        print(f"Company: {analysis.company_name}")
        print(f"\nPricing Tiers: {len(analysis.pricing_tiers)}")
        for tier in analysis.pricing_tiers:
            print(f"  - {tier}")
        
        print(f"\nKey Features ({len(analysis.key_features)}):")
        for feature in analysis.key_features:
            print(f"  - {feature}")
        
        print(f"\nMarket Positioning: {analysis.market_positioning}")
        print(f"\nStrengths: {', '.join(analysis.strengths)}")
        print(f"Weaknesses: {', '.join(analysis.weaknesses)}")


async def deep_research_example():
    """Example 4: Multi-round deep research with expanded queries."""
    print("\n=== Example 4: Deep Research with Multiple Rounds ===\n")
    
    async with Spiderweb(llm_client=GlueLLM()) as web:
        result = await web.research_and_report(
            persona="You are a tech journalist",
            instructions="Research latest developments in AI safety research, including recent papers and policy discussions",
            research_config=ResearchAgentConfig(
                num_queries=4,
                max_parallel_crawls=2,  # Lower concurrency for deeper research
            ),
            depth_config=SearchDepthConfig(
                max_search_rounds=2,
                crawl_results_per_round=5,
                when_to_go_deeper="expand_queries",
                num_expanded_queries=2,
            ),
        )
        
        print(f"Generated {len(result.queries_used)} initial queries")
        print(f"Total URLs crawled: {sum(len(t.get_all_urls()) for t in result.traces)}")
        print(f"\nReport preview ({len(result.report)} chars):")
        print("=" * 80)
        print(result.report[:800] + "..." if len(result.report) > 800 else result.report)
        print("=" * 80)


async def save_traces_example():
    """Example 5: Saving traces for later analysis."""
    print("\n=== Example 5: Saving Research Traces ===\n")
    
    async with Spiderweb(llm_client=GlueLLM()) as web:
        result = await web.research_and_report(
            persona="You are a research analyst",
            instructions="Research renewable energy trends in 2024",
            research_config=ResearchAgentConfig(num_queries=3),
            save_trace_to="./research_traces",
            trace_format="json",
            save_to="./crawled_content",
        )
        
        print(f"Saved {len(result.traces)} trace files to ./research_traces/")
        print(f"Saved crawled content to ./crawled_content/")
        print(f"\nQueries used: {result.queries_used}")


async def main():
    """Run all examples."""
    print("Spiderweb Research Agent Examples")
    print("=" * 80)
    
    # Run examples (comment out ones you don't want to run)
    await persona_driven_research_example()
    # await goal_driven_research_example()
    # await structured_report_example()
    # await deep_research_example()
    # await save_traces_example()
    
    print("\n" + "=" * 80)
    print("Examples complete!")


if __name__ == "__main__":
    asyncio.run(main())

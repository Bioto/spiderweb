"""Research agent module for persona-driven and goal-driven research workflows.

Provides query generation, report synthesis, and planning capabilities for
multi-crawler research workflows.
"""

from spiderweb.research.agent import (
    aggregate_traces_for_report,
    create_research_plan,
    decide_additional_queries,
    decide_additional_queries_with_reasoning,
    generate_research_queries,
    spill_trace_content_to_dir,
    spill_trace_to_store,
    summarize_content_batch,
    summarize_traces_in_batches,
    synthesize_report,
    synthesize_report_from_summaries,
)
from spiderweb.research.models import (
    ExpansionDecision,
    GoalResult,
    ResearchPlan,
    ResearchReportResult,
)
from spiderweb.research.storage import MarkdownResearchStore, ResearchContentStore

__all__ = [
    "aggregate_traces_for_report",
    "create_research_plan",
    "decide_additional_queries",
    "decide_additional_queries_with_reasoning",
    "generate_research_queries",
    "spill_trace_content_to_dir",
    "spill_trace_to_store",
    "summarize_content_batch",
    "summarize_traces_in_batches",
    "synthesize_report",
    "synthesize_report_from_summaries",
    "ExpansionDecision",
    "GoalResult",
    "MarkdownResearchStore",
    "ResearchContentStore",
    "ResearchPlan",
    "ResearchReportResult",
]

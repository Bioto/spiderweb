"""Built-in workflows that combine multiple Spiderweb operations.

Workflows provide default pipelines for common patterns (e.g. X search
then expand to posting users' followers/following) so you can run them
in one call without scripting the steps yourself.
"""

from spiderweb.workflows.x_search_expand import (
    XSearchExpandResult,
    XSearchExpandWorkflow,
)

__all__ = [
    "XSearchExpandResult",
    "XSearchExpandWorkflow",
]

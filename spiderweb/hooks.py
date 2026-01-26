"""Pipeline hook system for extensible processing.

Provides a callback mechanism for intercepting and modifying data at
key points in the document processing pipeline.

Example:
    Register a simple logging hook::

        from spiderweb import hooks, HookPoint

        def log_chunks(ctx):
            print(f"Created {len(ctx.data)} chunks")
            return ctx

        hooks.register(HookPoint.AFTER_CHUNK, log_chunks)

    Modify data in a hook::

        from spiderweb import hooks, HookPoint, HookContext

        def add_metadata(ctx: HookContext) -> HookContext:
            chunks = ctx.data
            for chunk in chunks:
                chunk.metadata.extra["processed"] = True
            ctx.modified_data = chunks
            return ctx

        hooks.register(HookPoint.AFTER_CHUNK, add_metadata)

    Skip an operation::

        def skip_large_files(ctx: HookContext) -> HookContext:
            from pathlib import Path
            if Path(ctx.data).stat().st_size > 100_000_000:
                ctx.skip = True
            return ctx

        hooks.register(HookPoint.BEFORE_EXTRACT, skip_large_files)
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Union


class HookPoint(str, Enum):
    """Pipeline hook points where callbacks can be registered.

    Each hook point corresponds to a specific stage in the pipeline:

    - BEFORE_EXTRACT: Called before file content extraction
    - AFTER_EXTRACT: Called after file content extraction
    - BEFORE_CHUNK: Called before document chunking
    - AFTER_CHUNK: Called after document chunking
    - BEFORE_CRAWL: Called before web crawling
    - AFTER_CRAWL: Called after web crawling
    """

    # Extraction hooks
    BEFORE_EXTRACT = "before_extract"
    AFTER_EXTRACT = "after_extract"

    # Chunking hooks
    BEFORE_CHUNK = "before_chunk"
    AFTER_CHUNK = "after_chunk"

    # Crawling hooks
    BEFORE_CRAWL = "before_crawl"
    AFTER_CRAWL = "after_crawl"


@dataclass
class HookContext:
    """Context passed to hook callbacks.

    Contains the data being processed and metadata about the operation.
    Hooks can modify the data or signal that the operation should be skipped.

    Attributes:
        hook_point: The hook point where this context was created
        data: The data being processed (Document, list[Chunk], CrawlResult, etc.)
        metadata: Additional context from the pipeline (document, config, etc.)
        skip: If True, the operation will be skipped
        modified_data: If set, this replaces the original data

    Example:
        >>> def my_hook(ctx: HookContext) -> HookContext:
        ...     # Modify the data
        ...     ctx.modified_data = transform(ctx.data)
        ...     return ctx
    """

    hook_point: HookPoint
    data: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    # Control flags
    skip: bool = False
    modified_data: Any = None


# Type aliases for hook callables
SyncHook = Callable[[HookContext], Union[HookContext, None]]
AsyncHook = Callable[[HookContext], Awaitable[Union[HookContext, None]]]
Hook = Union[SyncHook, AsyncHook]


class HookManager:
    """Manages registration and execution of pipeline hooks.

    Supports both synchronous and asynchronous hook callbacks.
    Hooks are executed in registration order for each hook point.

    Example:
        >>> manager = HookManager()
        >>> manager.register(HookPoint.AFTER_CHUNK, my_hook)
        >>> ctx = await manager.run(HookPoint.AFTER_CHUNK, chunks, document=doc)
        >>> result = ctx.modified_data or ctx.data
    """

    def __init__(self) -> None:
        """Initialize the hook manager with empty hook lists."""
        self._hooks: dict[HookPoint, list[Hook]] = {hp: [] for hp in HookPoint}

    def register(self, hook_point: HookPoint, callback: Hook) -> None:
        """Register a hook callback for a specific hook point.

        Hooks are called in the order they are registered.
        Both sync and async callbacks are supported.

        Args:
            hook_point: The pipeline stage to hook into
            callback: Function to call at this hook point

        Example:
            >>> def log_before_chunk(ctx):
            ...     print(f"Chunking document: {ctx.metadata.get('file_path')}")
            ...     return ctx
            >>> hooks.register(HookPoint.BEFORE_CHUNK, log_before_chunk)
        """
        self._hooks[hook_point].append(callback)

    def unregister(self, hook_point: HookPoint, callback: Hook) -> None:
        """Remove a previously registered hook callback.

        Args:
            hook_point: The hook point the callback was registered for
            callback: The callback function to remove

        Raises:
            ValueError: If the callback is not registered
        """
        self._hooks[hook_point].remove(callback)

    async def run(self, hook_point: HookPoint, data: Any, **metadata: Any) -> HookContext:
        """Execute all hooks registered for a hook point.

        Creates a HookContext with the provided data and metadata,
        then runs each registered hook in order. Hooks can:

        - Modify data by setting ctx.modified_data
        - Skip the operation by setting ctx.skip = True
        - Return a new context or None (keeps current context)

        If a hook sets skip=True, remaining hooks are not executed.

        Args:
            hook_point: The hook point to execute
            data: The data to pass to hooks (Document, chunks, URL, etc.)
            **metadata: Additional context (document, config, file_path, etc.)

        Returns:
            The final HookContext after all hooks have run

        Example:
            >>> ctx = await hooks.run(HookPoint.BEFORE_CHUNK, document, file_path="doc.pdf")
            >>> if ctx.skip:
            ...     return  # Operation was skipped
            >>> doc = ctx.modified_data or ctx.data
        """
        context = HookContext(hook_point=hook_point, data=data, metadata=metadata)

        for hook in self._hooks[hook_point]:
            result = hook(context)

            # Handle async hooks
            if asyncio.iscoroutine(result):
                result = await result

            # Update context if hook returned a new one
            if result is not None:
                context = result

            # Stop processing if skip was set
            if context.skip:
                break

        return context

    def clear(self, hook_point: HookPoint | None = None) -> None:
        """Clear registered hooks.

        Args:
            hook_point: If provided, clear only hooks for this point.
                If None, clear all hooks.

        Example:
            >>> hooks.clear(HookPoint.BEFORE_CHUNK)  # Clear specific point
            >>> hooks.clear()  # Clear all hooks
        """
        if hook_point is not None:
            self._hooks[hook_point] = []
        else:
            self._hooks = {hp: [] for hp in HookPoint}

    def has_hooks(self, hook_point: HookPoint) -> bool:
        """Check if any hooks are registered for a hook point.

        Args:
            hook_point: The hook point to check

        Returns:
            True if at least one hook is registered
        """
        return len(self._hooks[hook_point]) > 0

    def __repr__(self) -> str:
        """Return string representation showing hook counts."""
        counts = {hp.value: len(hooks) for hp, hooks in self._hooks.items() if hooks}
        return f"HookManager({counts})"


# Global hook manager instance
hooks = HookManager()

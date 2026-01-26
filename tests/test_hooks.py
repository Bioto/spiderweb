"""Tests for the pipeline hook system."""

import asyncio

import pytest

from spiderweb.hooks import HookContext, HookManager, HookPoint
from spiderweb.models.document import Document, DocumentMetadata


def _make_document(content: str = "Test content") -> Document:
    """Create a test document."""
    return Document(
        raw_content=content,
        markdown_content=content,
        metadata=DocumentMetadata(
            source="test://memory",
            file_type="md",
            extraction_method="test",
        ),
    )


class TestHookPoint:
    """Tests for HookPoint enum."""

    def test_hook_points_exist(self):
        """All expected hook points are defined."""
        assert HookPoint.BEFORE_EXTRACT == "before_extract"
        assert HookPoint.AFTER_EXTRACT == "after_extract"
        assert HookPoint.BEFORE_CHUNK == "before_chunk"
        assert HookPoint.AFTER_CHUNK == "after_chunk"
        assert HookPoint.BEFORE_CRAWL == "before_crawl"
        assert HookPoint.AFTER_CRAWL == "after_crawl"

    def test_hook_point_is_string_enum(self):
        """HookPoint values are strings."""
        for hp in HookPoint:
            assert isinstance(hp.value, str)


class TestHookContext:
    """Tests for HookContext dataclass."""

    def test_context_creation(self):
        """Create a hook context with data."""
        doc = _make_document()
        ctx = HookContext(
            hook_point=HookPoint.BEFORE_CHUNK,
            data=doc,
            metadata={"source": "test"},
        )

        assert ctx.hook_point == HookPoint.BEFORE_CHUNK
        assert ctx.data is doc
        assert ctx.metadata["source"] == "test"
        assert ctx.skip is False
        assert ctx.modified_data is None

    def test_context_default_metadata(self):
        """Context has empty metadata dict by default."""
        ctx = HookContext(hook_point=HookPoint.BEFORE_CRAWL, data="url")
        assert ctx.metadata == {}

    def test_context_skip_flag(self):
        """Context skip flag can be set to skip operations."""
        ctx = HookContext(hook_point=HookPoint.BEFORE_CRAWL, data="url")
        ctx.skip = True

        assert ctx.skip is True

    def test_context_modified_data(self):
        """Hooks can provide modified data."""
        ctx = HookContext(hook_point=HookPoint.AFTER_CHUNK, data=[])
        ctx.modified_data = ["modified", "chunks"]

        assert ctx.modified_data == ["modified", "chunks"]


class TestHookManager:
    """Tests for HookManager class."""

    def test_register_sync_hook(self):
        """Register a synchronous hook."""
        manager = HookManager()
        called = []

        def my_hook(ctx: HookContext) -> HookContext:
            called.append(ctx.hook_point)
            return ctx

        manager.register(HookPoint.BEFORE_CHUNK, my_hook)

        ctx = asyncio.run(manager.run(HookPoint.BEFORE_CHUNK, "data"))

        assert called == [HookPoint.BEFORE_CHUNK]
        assert ctx.data == "data"

    @pytest.mark.asyncio
    async def test_register_async_hook(self):
        """Register and execute an asynchronous hook."""
        manager = HookManager()
        called = []

        async def my_async_hook(ctx: HookContext) -> HookContext:
            await asyncio.sleep(0.01)
            called.append("async")
            return ctx

        manager.register(HookPoint.AFTER_CRAWL, my_async_hook)

        await manager.run(HookPoint.AFTER_CRAWL, "result")

        assert called == ["async"]

    @pytest.mark.asyncio
    async def test_hooks_run_in_order(self):
        """Multiple hooks run in registration order."""
        manager = HookManager()
        order = []

        def hook_a(ctx: HookContext) -> HookContext:
            order.append("a")
            return ctx

        def hook_b(ctx: HookContext) -> HookContext:
            order.append("b")
            return ctx

        def hook_c(ctx: HookContext) -> HookContext:
            order.append("c")
            return ctx

        manager.register(HookPoint.BEFORE_EXTRACT, hook_a)
        manager.register(HookPoint.BEFORE_EXTRACT, hook_b)
        manager.register(HookPoint.BEFORE_EXTRACT, hook_c)

        await manager.run(HookPoint.BEFORE_EXTRACT, "file.pdf")

        assert order == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_hook_can_modify_data(self):
        """Hook can modify context data."""
        manager = HookManager()

        def transform_hook(ctx: HookContext) -> HookContext:
            ctx.modified_data = ctx.data.upper()
            return ctx

        manager.register(HookPoint.BEFORE_CRAWL, transform_hook)

        ctx = await manager.run(HookPoint.BEFORE_CRAWL, "https://example.com")

        assert ctx.modified_data == "HTTPS://EXAMPLE.COM"

    @pytest.mark.asyncio
    async def test_hook_skip_stops_chain(self):
        """Setting skip=True stops further hooks."""
        manager = HookManager()
        order = []

        def hook_that_skips(ctx: HookContext) -> HookContext:
            order.append("skipper")
            ctx.skip = True
            return ctx

        def hook_after(ctx: HookContext) -> HookContext:
            order.append("should-not-run")
            return ctx

        manager.register(HookPoint.BEFORE_CHUNK, hook_that_skips)
        manager.register(HookPoint.BEFORE_CHUNK, hook_after)

        ctx = await manager.run(HookPoint.BEFORE_CHUNK, "doc")

        assert order == ["skipper"]
        assert ctx.skip is True

    @pytest.mark.asyncio
    async def test_metadata_passed_to_context(self):
        """Metadata kwargs are passed to hook context."""
        manager = HookManager()
        received_metadata = {}

        def capture_hook(ctx: HookContext) -> HookContext:
            received_metadata.update(ctx.metadata)
            return ctx

        manager.register(HookPoint.AFTER_CHUNK, capture_hook)

        await manager.run(
            HookPoint.AFTER_CHUNK,
            [],
            document="doc",
            chunk_count=5,
            file_path="/path/to/file",
        )

        assert received_metadata["document"] == "doc"
        assert received_metadata["chunk_count"] == 5
        assert received_metadata["file_path"] == "/path/to/file"

    def test_unregister_hook(self):
        """Remove a registered hook."""
        manager = HookManager()

        def my_hook(ctx: HookContext) -> HookContext:
            return ctx

        manager.register(HookPoint.BEFORE_CRAWL, my_hook)
        assert manager.has_hooks(HookPoint.BEFORE_CRAWL)

        manager.unregister(HookPoint.BEFORE_CRAWL, my_hook)
        assert not manager.has_hooks(HookPoint.BEFORE_CRAWL)

    def test_unregister_unknown_hook_raises(self):
        """Unregistering non-existent hook raises ValueError."""
        manager = HookManager()

        def my_hook(ctx: HookContext) -> HookContext:
            return ctx

        with pytest.raises(ValueError):
            manager.unregister(HookPoint.BEFORE_CHUNK, my_hook)

    def test_clear_specific_hook_point(self):
        """Clear hooks for a specific hook point."""
        manager = HookManager()

        def hook(ctx: HookContext) -> HookContext:
            return ctx

        manager.register(HookPoint.BEFORE_CHUNK, hook)
        manager.register(HookPoint.AFTER_CHUNK, hook)

        manager.clear(HookPoint.BEFORE_CHUNK)

        assert not manager.has_hooks(HookPoint.BEFORE_CHUNK)
        assert manager.has_hooks(HookPoint.AFTER_CHUNK)

    def test_clear_all_hooks(self):
        """Clear all hooks."""
        manager = HookManager()

        def hook(ctx: HookContext) -> HookContext:
            return ctx

        for hp in HookPoint:
            manager.register(hp, hook)

        manager.clear()

        for hp in HookPoint:
            assert not manager.has_hooks(hp)

    def test_has_hooks(self):
        """Check if hooks are registered."""
        manager = HookManager()

        assert not manager.has_hooks(HookPoint.BEFORE_CHUNK)

        def hook(ctx: HookContext) -> HookContext:
            return ctx

        manager.register(HookPoint.BEFORE_CHUNK, hook)
        assert manager.has_hooks(HookPoint.BEFORE_CHUNK)

    def test_repr(self):
        """String representation shows hook counts."""
        manager = HookManager()

        def hook(ctx: HookContext) -> HookContext:
            return ctx

        manager.register(HookPoint.BEFORE_CHUNK, hook)
        manager.register(HookPoint.BEFORE_CHUNK, hook)
        manager.register(HookPoint.AFTER_CRAWL, hook)

        repr_str = repr(manager)
        assert "HookManager" in repr_str
        assert "before_chunk" in repr_str
        assert "after_crawl" in repr_str

    @pytest.mark.asyncio
    async def test_hook_returning_none_keeps_context(self):
        """Hook returning None doesn't replace context."""
        manager = HookManager()
        original_data = "original"

        def hook_returns_none(ctx: HookContext) -> None:
            # Doesn't return anything
            pass

        manager.register(HookPoint.BEFORE_CRAWL, hook_returns_none)

        ctx = await manager.run(HookPoint.BEFORE_CRAWL, original_data)

        assert ctx.data == original_data

    @pytest.mark.asyncio
    async def test_mixed_sync_async_hooks(self):
        """Can mix sync and async hooks."""
        manager = HookManager()
        order = []

        def sync_hook(ctx: HookContext) -> HookContext:
            order.append("sync")
            return ctx

        async def async_hook(ctx: HookContext) -> HookContext:
            await asyncio.sleep(0.01)
            order.append("async")
            return ctx

        manager.register(HookPoint.BEFORE_EXTRACT, sync_hook)
        manager.register(HookPoint.BEFORE_EXTRACT, async_hook)
        manager.register(HookPoint.BEFORE_EXTRACT, sync_hook)

        await manager.run(HookPoint.BEFORE_EXTRACT, "data")

        assert order == ["sync", "async", "sync"]


class TestHookIntegration:
    """Integration tests for hooks with real data types."""

    @pytest.mark.asyncio
    async def test_before_chunk_can_modify_document(self):
        """BEFORE_CHUNK hook can modify document before chunking."""
        manager = HookManager()

        def add_prefix(ctx: HookContext) -> HookContext:
            doc = ctx.data
            modified = Document(
                raw_content="[PREFIX] " + doc.raw_content,
                markdown_content="[PREFIX] " + doc.markdown_content,
                metadata=doc.metadata,
            )
            ctx.modified_data = modified
            return ctx

        manager.register(HookPoint.BEFORE_CHUNK, add_prefix)

        original = _make_document("Original content")
        ctx = await manager.run(HookPoint.BEFORE_CHUNK, original)

        result = ctx.modified_data or ctx.data
        assert result.markdown_content.startswith("[PREFIX]")
        assert "Original content" in result.markdown_content

    @pytest.mark.asyncio
    async def test_after_chunk_can_filter_chunks(self):
        """AFTER_CHUNK hook can filter or modify chunks."""
        manager = HookManager()

        def filter_short_chunks(ctx: HookContext) -> HookContext:
            chunks = ctx.data
            ctx.modified_data = [c for c in chunks if len(c) > 10]
            return ctx

        manager.register(HookPoint.AFTER_CHUNK, filter_short_chunks)

        chunks = ["short", "this is a longer chunk that passes", "tiny"]
        ctx = await manager.run(HookPoint.AFTER_CHUNK, chunks)

        result = ctx.modified_data
        assert len(result) == 1
        assert "longer" in result[0]

    @pytest.mark.asyncio
    async def test_before_crawl_can_rewrite_url(self):
        """BEFORE_CRAWL hook can modify URL."""
        manager = HookManager()

        def add_tracking(ctx: HookContext) -> HookContext:
            url = ctx.data
            if "?" in url:
                ctx.modified_data = url + "&source=spiderweb"
            else:
                ctx.modified_data = url + "?source=spiderweb"
            return ctx

        manager.register(HookPoint.BEFORE_CRAWL, add_tracking)

        ctx = await manager.run(HookPoint.BEFORE_CRAWL, "https://example.com")
        assert ctx.modified_data == "https://example.com?source=spiderweb"

        ctx2 = await manager.run(HookPoint.BEFORE_CRAWL, "https://example.com?page=1")
        assert ctx2.modified_data == "https://example.com?page=1&source=spiderweb"

    @pytest.mark.asyncio
    async def test_skip_prevents_operation(self):
        """Hooks can prevent operations by setting skip=True."""
        manager = HookManager()
        
        blocked_domains = ["blocked.com", "spam.org"]

        def block_domains(ctx: HookContext) -> HookContext:
            url = ctx.data
            for domain in blocked_domains:
                if domain in url:
                    ctx.skip = True
                    break
            return ctx

        manager.register(HookPoint.BEFORE_CRAWL, block_domains)

        # Allowed URL
        ctx = await manager.run(HookPoint.BEFORE_CRAWL, "https://allowed.com")
        assert not ctx.skip

        # Blocked URL
        ctx = await manager.run(HookPoint.BEFORE_CRAWL, "https://blocked.com/page")
        assert ctx.skip

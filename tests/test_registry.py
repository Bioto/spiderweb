"""Tests for the component registry system."""

import pytest

from spiderweb.chunkers import Chunker, HierarchicalChunker, SentenceChunker
from spiderweb.models.document import Chunk, Document, DocumentMetadata
from spiderweb.registry import (
    ComponentRegistry,
    chunker_registry,
    crawler_registry,
    extractor_registry,
    search_provider_registry,
)


def _make_document(markdown: str = "Test content") -> Document:
    """Create a test document."""
    return Document(
        raw_content=markdown,
        markdown_content=markdown,
        metadata=DocumentMetadata(
            source="memory://test",
            file_type="md",
            extraction_method="test",
        ),
    )


class TestComponentRegistry:
    """Tests for the ComponentRegistry class."""

    def test_register_and_get_class(self):
        """Register a class and retrieve it by name."""
        registry: ComponentRegistry[Chunker] = ComponentRegistry("chunker")
        registry.register("test-chunker", HierarchicalChunker)

        retrieved = registry.get("test-chunker")
        assert retrieved is HierarchicalChunker

    def test_register_factory_function(self):
        """Register a factory function that creates components."""
        registry: ComponentRegistry[Chunker] = ComponentRegistry("chunker")

        def create_chunker(max_size: int = 500) -> HierarchicalChunker:
            return HierarchicalChunker(max_chunk_size=max_size)

        registry.register_factory("factory-chunker", create_chunker)

        chunker = registry.create("factory-chunker", max_size=200)
        assert isinstance(chunker, HierarchicalChunker)
        assert chunker.max_chunk_size == 200

    def test_create_with_kwargs(self):
        """Create instance with keyword arguments."""
        registry: ComponentRegistry[Chunker] = ComponentRegistry("chunker")
        registry.register("hierarchical", HierarchicalChunker)

        chunker = registry.create("hierarchical", max_chunk_size=500)
        assert chunker.max_chunk_size == 500

    def test_get_unknown_raises_keyerror(self):
        """Accessing unknown component raises KeyError with helpful message."""
        registry: ComponentRegistry[Chunker] = ComponentRegistry("chunker")
        registry.register("known", HierarchicalChunker)

        with pytest.raises(KeyError) as exc_info:
            registry.get("unknown")

        error_message = str(exc_info.value)
        assert "unknown" in error_message
        assert "known" in error_message
        assert "chunker" in error_message.lower()

    def test_list_registered(self):
        """List all registered component names."""
        registry: ComponentRegistry[Chunker] = ComponentRegistry("chunker")
        registry.register("alpha", HierarchicalChunker)
        registry.register("beta", SentenceChunker)

        names = registry.list()
        assert set(names) == {"alpha", "beta"}

    def test_list_is_sorted(self):
        """List returns sorted names."""
        registry: ComponentRegistry[Chunker] = ComponentRegistry("chunker")
        registry.register("zebra", HierarchicalChunker)
        registry.register("apple", SentenceChunker)
        registry.register("mango", HierarchicalChunker)

        names = registry.list()
        assert names == ["apple", "mango", "zebra"]

    def test_contains(self):
        """Check if a component is registered."""
        registry: ComponentRegistry[Chunker] = ComponentRegistry("chunker")
        registry.register("exists", HierarchicalChunker)

        assert "exists" in registry
        assert "missing" not in registry

    def test_factory_takes_precedence(self):
        """Factory registration overrides class registration for get()."""
        registry: ComponentRegistry[Chunker] = ComponentRegistry("chunker")
        registry.register("test", HierarchicalChunker)
        registry.register_factory("test", lambda: "factory-result")

        result = registry.get("test")
        assert result() == "factory-result"

    def test_repr(self):
        """String representation shows registered components."""
        registry: ComponentRegistry[Chunker] = ComponentRegistry("chunker")
        registry.register("a", HierarchicalChunker)
        registry.register("b", SentenceChunker)

        repr_str = repr(registry)
        assert "chunker" in repr_str
        assert "a" in repr_str
        assert "b" in repr_str

    def test_component_type_property(self):
        """Component type property returns the type name."""
        registry: ComponentRegistry[Chunker] = ComponentRegistry("my-type")
        assert registry.component_type == "my-type"


class TestDefaultRegistrations:
    """Verify built-in components are registered by default."""

    def test_chunker_defaults_registered(self):
        """All default chunkers are registered."""
        assert "hierarchical" in chunker_registry
        assert "sentence" in chunker_registry
        assert "semantic" in chunker_registry
        assert "sliding_window" in chunker_registry

    def test_crawler_defaults_registered(self):
        """All default crawlers are registered."""
        assert "http" in crawler_registry
        assert "crawl4ai" in crawler_registry
        assert "firecrawl" in crawler_registry
        assert "x" in crawler_registry

    def test_search_provider_defaults_registered(self):
        """Built-in search providers are registered."""
        assert "duckduckgo" in search_provider_registry
        assert "stub" in search_provider_registry
        assert "firecrawl" in search_provider_registry

    def test_extractor_defaults_registered(self):
        """Default extractors are registered."""
        assert "markitdown" in extractor_registry

    def test_can_create_default_chunker(self):
        """Can create a default chunker from registry."""
        chunker = chunker_registry.create("hierarchical", max_chunk_size=500)
        assert chunker.max_chunk_size == 500

        doc = _make_document("# Title\n\nSome content here.")
        chunks = chunker.chunk(doc)
        assert isinstance(chunks, list)


class TestCustomComponentRegistration:
    """Test registering custom components."""

    def test_register_custom_chunker(self):
        """External developers can register custom chunkers."""
        # Create a separate registry to avoid polluting the global one
        test_registry: ComponentRegistry[Chunker] = ComponentRegistry("chunker")

        class MyCustomChunker:
            """A simple custom chunker for testing."""

            def chunk(self, document: Document) -> list[Chunk]:
                # Just return empty list for testing
                return []

        test_registry.register("my-custom", MyCustomChunker)

        assert "my-custom" in test_registry
        instance = test_registry.create("my-custom")
        assert isinstance(instance, MyCustomChunker)

        # Verify it implements the protocol
        doc = _make_document("Test content")
        result = instance.chunk(doc)
        assert result == []

    def test_register_chunker_with_from_config(self):
        """Custom chunker with from_config classmethod."""
        from spiderweb.models.config import ChunkerConfig

        test_registry: ComponentRegistry[Chunker] = ComponentRegistry("chunker")

        class ConfigurableChunker:
            """Custom chunker that supports from_config."""

            def __init__(self, max_size: int = 1000):
                self.max_size = max_size

            def chunk(self, document: Document) -> list[Chunk]:
                return []

            @classmethod
            def from_config(cls, config: ChunkerConfig) -> "ConfigurableChunker":
                return cls(max_size=config.max_chunk_size)

        test_registry.register("configurable", ConfigurableChunker)

        # Create via registry with kwargs
        instance = test_registry.create("configurable", max_size=500)
        assert instance.max_size == 500

        # Create via from_config
        config = ChunkerConfig(max_chunk_size=750)
        chunker_cls = test_registry.get("configurable")
        instance2 = chunker_cls.from_config(config)
        assert instance2.max_size == 750

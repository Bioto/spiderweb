"""Component registry for extensible pipeline components.

Provides a generic registry pattern for registering and retrieving
chunkers, crawlers, extractors, and other pipeline components by name.

Example:
    Register a custom chunker::

        from spiderweb import chunker_registry
        from spiderweb.models.document import Document, Chunk

        class MyCustomChunker:
            def chunk(self, document: Document) -> list[Chunk]:
                # Custom chunking logic
                ...

        chunker_registry.register("my-chunker", MyCustomChunker)

    Use via config::

        from spiderweb import Spiderweb
        from spiderweb.models.config import ChunkerConfig

        config = ChunkerConfig(strategy="my-chunker")
        web = Spiderweb(chunker_config=config)

    Register a factory function::

        def create_chunker_with_options(**kwargs):
            return MyCustomChunker(**kwargs)

        chunker_registry.register_factory("my-chunker-factory", create_chunker_with_options)
"""

from typing import TYPE_CHECKING, Callable, Generic, TypeVar

if TYPE_CHECKING:
    from spiderweb.chunkers.base import Chunker
    from spiderweb.crawlers.base import Crawler
    from spiderweb.extractors.base import Extractor
    from spiderweb.search.base import SearchProvider

T = TypeVar("T")


class ComponentRegistry(Generic[T]):
    """Generic registry for pipeline components.

    Allows registering component classes or factory functions by name,
    then retrieving or instantiating them later. Supports both direct
    class registration and factory functions for complex initialization.

    Type Parameters:
        T: The component protocol/type this registry manages

    Attributes:
        component_type: Human-readable name for error messages

    Example:
        >>> registry = ComponentRegistry[Chunker]("chunker")
        >>> registry.register("custom", MyChunker)
        >>> chunker = registry.create("custom", max_size=500)
    """

    def __init__(self, component_type: str) -> None:
        """Initialize a component registry.

        Args:
            component_type: Human-readable name for this component type
                (used in error messages, e.g., "chunker", "crawler")
        """
        self._registry: dict[str, type[T]] = {}
        self._factories: dict[str, Callable[..., T]] = {}
        self._component_type = component_type

    @property
    def component_type(self) -> str:
        """Human-readable component type name."""
        return self._component_type

    def register(self, name: str, cls: type[T]) -> None:
        """Register a component class by name.

        The class must implement the appropriate protocol for this registry
        (Chunker, Crawler, or Extractor).

        Args:
            name: Unique name to register the component under
            cls: The component class to register

        Example:
            >>> chunker_registry.register("paragraph", ParagraphChunker)
        """
        self._registry[name] = cls

    def register_factory(self, name: str, factory: Callable[..., T]) -> None:
        """Register a factory function that creates the component.

        Use this for components that require complex initialization,
        dependency injection, or runtime configuration.

        Args:
            name: Unique name to register the factory under
            factory: Callable that returns a component instance

        Example:
            >>> def create_semantic_chunker(llm_client, threshold=0.7):
            ...     return SemanticChunker(llm_client, threshold)
            >>> chunker_registry.register_factory("semantic-llm", create_semantic_chunker)
        """
        self._factories[name] = factory

    def get(self, name: str) -> type[T] | Callable[..., T]:
        """Get a registered component class or factory by name.

        Factories take precedence over classes if both are registered
        with the same name.

        Args:
            name: The registered name of the component

        Returns:
            The registered class or factory callable

        Raises:
            KeyError: If no component is registered with that name
        """
        if name in self._factories:
            return self._factories[name]
        if name in self._registry:
            return self._registry[name]
        available = self.list()
        raise KeyError(
            f"Unknown {self._component_type}: '{name}'. "
            f"Available: {available}"
        )

    def create(self, name: str, **kwargs: object) -> T:
        """Create an instance of a registered component.

        Retrieves the component class or factory and instantiates it
        with the provided keyword arguments.

        Args:
            name: The registered name of the component
            **kwargs: Arguments to pass to the constructor or factory

        Returns:
            A new instance of the component

        Raises:
            KeyError: If no component is registered with that name

        Example:
            >>> chunker = chunker_registry.create("hierarchical", max_chunk_size=500)
        """
        component = self.get(name)
        return component(**kwargs)

    def list(self) -> list[str]:
        """List all registered component names.

        Returns:
            Sorted list of registered component names
        """
        return sorted(set(self._registry.keys()) | set(self._factories.keys()))

    def __contains__(self, name: str) -> bool:
        """Check if a component is registered.

        Args:
            name: The name to check

        Returns:
            True if a component is registered with that name
        """
        return name in self._registry or name in self._factories

    def __repr__(self) -> str:
        """Return string representation of the registry."""
        return f"ComponentRegistry[{self._component_type}]({self.list()})"


# Global registries - populated by each package's __init__.py
chunker_registry: ComponentRegistry["Chunker"] = ComponentRegistry("chunker")
crawler_registry: ComponentRegistry["Crawler"] = ComponentRegistry("crawler")
extractor_registry: ComponentRegistry["Extractor"] = ComponentRegistry("extractor")
search_provider_registry: ComponentRegistry["SearchProvider"] = ComponentRegistry("search_provider")

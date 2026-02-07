"""Chunk add-ons for enriching chunks with additional processing.

Chunk add-ons run after chunking and can attach data to chunks via
chunk.metadata.extra. They are registered in chunk_addon_registry and
enabled via ChunkAddOnConfig.

Example:
    Register a custom add-on::

        from spiderweb.addons.base import ChunkAddOn
        from spiderweb import chunk_addon_registry

        class MyAddOn(ChunkAddOn):
            async def process(self, chunks, **kwargs):
                for chunk in chunks:
                    chunk.metadata.extra["my_data"] = "value"
                return chunks

        chunk_addon_registry.register("my-addon", MyAddOn)

    Enable via config::

        from spiderweb import Spiderweb
        from spiderweb.models.config import ChunkAddOnConfig

        config = ChunkAddOnConfig(enabled=["my-addon"])
        web = Spiderweb(chunk_addon_config=config)
"""

from typing import Any

from spiderweb.addons.base import ChunkAddOn
from spiderweb.addons.facts import FactsAddOn
from spiderweb.registry import chunk_addon_registry

__all__ = ["ChunkAddOn", "FactsAddOn", "chunk_addon_registry"]

# Register built-in add-ons
def _register_builtin_addons() -> None:
    """Register built-in chunk add-ons."""
    # Facts add-on requires llm_client, so we register a factory
    def create_facts_addon(llm_client: Any | None = None, **kwargs: object) -> FactsAddOn:
        """Factory for creating FactsAddOn instances."""
        return FactsAddOn(llm_client=llm_client, **kwargs)
    
    chunk_addon_registry.register_factory("facts", create_facts_addon)

_register_builtin_addons()

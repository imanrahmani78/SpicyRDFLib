from __future__ import annotations
from rdflib import Graph
from .index import VectorIndex
from .functions import set_active_index, register_functions


class SpicyGraph(Graph):
    """Graph subclass with vector-augmented SPARQL support."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vector_index: VectorIndex | None = None
        self._vector_enabled: bool = False

    def enable_vector_search(self) -> None:
        """Build the embedding index and register cosine SPARQL extension functions."""
        idx = VectorIndex()
        idx.build(self)
        self._vector_index = idx
        set_active_index(idx)
        register_functions()
        self._vector_enabled = True

    def add(self, triple):
        super().add(triple)
        if self._vector_enabled and self._vector_index is not None:
            self._vector_index.add_triple(*triple)

    def parse(self, *args, **kwargs):
        result = super().parse(*args, **kwargs)
        if self._vector_enabled:
            self.enable_vector_search()
        return result

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
        self._store: object = None  # LanceStore instance when persistence is on

    def enable_vector_search(self, index_path: str | None = None) -> None:
        """Build the embedding index and register cosine SPARQL extension functions.

        Parameters
        ----------
        index_path:
            Optional directory path for a LanceDB persistent index.  When
            supplied the index is loaded from disk if the fingerprint matches
            the current graph; otherwise it is built and saved.  Subsequent
            calls with the same path skip re-embedding entirely.
        """
        store = None
        if index_path is not None:
            from .store import LanceStore
            store = LanceStore(index_path)
            self._store = store

        idx = VectorIndex()
        idx.build(self, store=store)
        self._vector_index = idx
        set_active_index(idx)
        register_functions()
        self._vector_enabled = True

    def add(self, triple):
        super().add(triple)
        if self._vector_enabled and self._vector_index is not None:
            self._vector_index.add_triple(*triple, store=self._store)

    def parse(self, *args, **kwargs):
        result = super().parse(*args, **kwargs)
        if self._vector_enabled:
            self.enable_vector_search(
                index_path=self._store.path if self._store else None
            )
        return result

from __future__ import annotations
import numpy as np
from rdflib import Graph, Literal
from .embedder import embed, batch_embed

_EMBED_DIM = 384  # all-MiniLM-L6-v2 output dimension


def _term_text(term) -> str:
    if isinstance(term, Literal):
        return str(term)
    uri = str(term)
    for sep in ("#", "/"):
        idx = uri.rfind(sep)
        if idx != -1 and idx < len(uri) - 1:
            return uri[idx + 1:]
    return uri


class VectorIndex:
    def __init__(self):
        self.subject_index: dict[str, np.ndarray] = {}
        self.predicate_index: dict[str, np.ndarray] = {}
        self.object_index: dict[str, np.ndarray] = {}

        # Per-component stacked matrices (N × 384) for one-shot matmul scoring
        self._matrices: dict[str, np.ndarray] = {}
        self._keys: dict[str, list[str]] = {}

        # Score cache: (component, query_text) -> {term_key: score}
        # Cleared whenever the index changes so results stay fresh.
        self._score_cache: dict[tuple[str, str], dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Build / update
    # ------------------------------------------------------------------

    def build(self, graph: Graph) -> None:
        subjects: set = set()
        predicates: set = set()
        objects: set = set()
        for s, p, o in graph:
            subjects.add(s)
            predicates.add(p)
            objects.add(o)

        for terms, idx in (
            (subjects, self.subject_index),
            (predicates, self.predicate_index),
            (objects, self.object_index),
        ):
            term_list = list(terms)
            texts = [_term_text(t) for t in term_list]
            vecs = batch_embed(texts)
            for term, vec in zip(term_list, vecs):
                idx[str(term)] = vec

        self._rebuild_matrices()

    def add_triple(self, s, p, o) -> None:
        changed = False
        for term, idx in (
            (s, self.subject_index),
            (p, self.predicate_index),
            (o, self.object_index),
        ):
            key = str(term)
            if key not in idx:
                idx[key] = embed(_term_text(term))
                changed = True
        if changed:
            self._rebuild_matrices()

    def _rebuild_matrices(self) -> None:
        self._score_cache.clear()
        for component, idx in (
            ("subject", self.subject_index),
            ("predicate", self.predicate_index),
            ("object", self.object_index),
        ):
            if idx:
                keys = list(idx.keys())
                mat = np.stack([idx[k] for k in keys]).astype(np.float32)
                mat.flags.writeable = False
                self._keys[component] = keys
                self._matrices[component] = mat
            else:
                self._keys[component] = []
                self._matrices[component] = np.empty((0, _EMBED_DIM), dtype=np.float32)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_all_scores(self, component: str, query_text: str) -> dict[str, float]:
        """Return cosine scores for every term in a component via one matmul.

        Result is cached per (component, query_text) and reused across FILTER rows.
        """
        cache_key = (component, query_text)
        if cache_key not in self._score_cache:
            mat = self._matrices.get(component)
            keys = self._keys.get(component, [])
            if mat is None or len(keys) == 0:
                self._score_cache[cache_key] = {}
            else:
                query_vec = embed(query_text)
                scores = (mat @ query_vec).tolist()
                self._score_cache[cache_key] = dict(zip(keys, scores))
        return self._score_cache[cache_key]

    def top_k(self, component: str, query_text: str, k: int) -> list[tuple[str, float]]:
        """Return the top-k (term_key, score) pairs for a component, descending."""
        scores = self.get_all_scores(component, query_text)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

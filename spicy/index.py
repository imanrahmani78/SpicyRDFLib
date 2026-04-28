
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from rdflib import Graph, Literal
from .embedder import embed, batch_embed, chunk_text

if TYPE_CHECKING:
    from .store import LanceStore

_EMBED_DIM = 384
_CHUNK_THRESHOLD = 300  # chars; literals longer than this are chunked


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
        # term_key -> [chunk_vec, ...]  (one entry for short terms, multiple for long literals)
        self._chunks: dict[str, dict[str, list[np.ndarray]]] = {
            "subject": {}, "predicate": {}, "object": {}
        }

        # Flattened structures for fast matmul scoring
        # _chunk_keys[component][i] == term_key for row i of _matrices[component]
        self._chunk_keys: dict[str, list[str]] = {}
        self._matrices: dict[str, np.ndarray] = {}

        # Score cache: (component, query_text) -> {term_key: score}
        self._score_cache: dict[tuple[str, str], dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Build / update
    # ------------------------------------------------------------------

    def build(self, graph: Graph, store: "LanceStore | None" = None) -> None:
        if store and store.is_valid(graph):
            self._load_from_store(store)
            return

        subjects: set = set()
        predicates: set = set()
        objects: set = set()
        for s, p, o in graph:
            subjects.add(s)
            predicates.add(p)
            objects.add(o)

        for component, terms in (
            ("subject", subjects),
            ("predicate", predicates),
            ("object", objects),
        ):
            self._embed_terms(component, list(terms))

        self._rebuild_matrices()

        if store:
            store.write(self._export_rows(), graph)

    def _embed_terms(self, component: str, terms: list) -> None:
        """Embed a list of RDF terms into self._chunks[component], skipping already-known terms."""
        idx = self._chunks[component]
        new_terms = [t for t in terms if str(t) not in idx]
        if not new_terms:
            return

        # Collect all chunk texts needed
        term_chunk_map: dict[str, list[str]] = {}
        all_chunk_texts: list[str] = []
        for term in new_terms:
            key = str(term)
            text = _term_text(term)
            chunks = chunk_text(text, _CHUNK_THRESHOLD)
            term_chunk_map[key] = chunks
            all_chunk_texts.extend(chunks)

        # Batch embed all unique chunk texts
        unique = list(dict.fromkeys(all_chunk_texts))  # deduplicate, preserve order
        vecs_list = batch_embed(unique)
        vec_map: dict[str, np.ndarray] = dict(zip(unique, vecs_list))

        for term in new_terms:
            key = str(term)
            idx[key] = [vec_map[c] for c in term_chunk_map[key]]

    def add_triple(self, s, p, o, store: "LanceStore | None" = None) -> None:
        new_rows: list[dict] = []
        changed = False
        for term, component in ((s, "subject"), (p, "predicate"), (o, "object")):
            key = str(term)
            if key not in self._chunks[component]:
                text = _term_text(term)
                chunks = chunk_text(text, _CHUNK_THRESHOLD)
                vecs = batch_embed(chunks)
                self._chunks[component][key] = vecs
                if store:
                    for ci, vec in enumerate(vecs):
                        new_rows.append({"component": component, "term_key": key, "chunk_idx": ci, "vector": vec})
                changed = True
        if changed:
            self._rebuild_matrices()
            if store and new_rows:
                store.append(new_rows)

    def _load_from_store(self, store: "LanceStore") -> None:
        """Populate _chunks from persisted rows without re-embedding."""
        rows = store.load_rows()
        for r in rows:
            comp = r["component"]
            key = r["term_key"]
            ci = r["chunk_idx"]
            vec = r["vector"]
            vec.flags.writeable = False
            chunks = self._chunks[comp].setdefault(key, [])
            # Ensure list is long enough
            while len(chunks) <= ci:
                chunks.append(None)  # type: ignore[arg-type]
            chunks[ci] = vec
        self._rebuild_matrices()

    def _export_rows(self) -> list[dict]:
        """Export all chunks as flat row dicts for LanceStore."""
        rows: list[dict] = []
        for component, idx in self._chunks.items():
            for term_key, vecs in idx.items():
                for ci, vec in enumerate(vecs):
                    rows.append({"component": component, "term_key": term_key, "chunk_idx": ci, "vector": vec})
        return rows

    def _rebuild_matrices(self) -> None:
        self._score_cache.clear()
        for component, idx in self._chunks.items():
            if not idx:
                self._chunk_keys[component] = []
                self._matrices[component] = np.empty((0, _EMBED_DIM), dtype=np.float32)
                continue
            keys: list[str] = []
            vecs: list[np.ndarray] = []
            for term_key, chunk_vecs in idx.items():
                for v in chunk_vecs:
                    if v is not None:
                        keys.append(term_key)
                        vecs.append(v)
            mat = np.stack(vecs).astype(np.float32)
            mat.flags.writeable = False
            self._chunk_keys[component] = keys
            self._matrices[component] = mat

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_all_scores(self, component: str, query_text: str) -> dict[str, float]:
        """Return per-term cosine scores via one matmul, aggregating chunks by max.

        Results are cached per (component, query_text) and reused across FILTER rows.
        """
        cache_key = (component, query_text)
        if cache_key in self._score_cache:
            return self._score_cache[cache_key]

        mat = self._matrices.get(component)
        keys = self._chunk_keys.get(component, [])
        if mat is None or len(keys) == 0:
            self._score_cache[cache_key] = {}
            return {}

        query_vec = embed(query_text).astype(np.float32)
        chunk_scores = (mat @ query_vec).tolist()

        # Aggregate: keep max score per unique term key
        scores: dict[str, float] = {}
        for key, s in zip(keys, chunk_scores):
            if s > scores.get(key, -1.0):
                scores[key] = s

        self._score_cache[cache_key] = scores
        return scores

    def top_k(self, component: str, query_text: str, k: int) -> list[tuple[str, float]]:
        """Return the top-k (term_key, score) pairs for a component, descending."""
        scores = self.get_all_scores(component, query_text)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

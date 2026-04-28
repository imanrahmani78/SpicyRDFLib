"""LanceDB-backed persistent vector store for SpicyRDFLib.

Stores per-chunk embeddings and provides ANN search via HNSW.
Used as an optional persistence layer — VectorIndex still owns the
in-memory scoring matrices; LanceStore only handles load/save.
"""
from __future__ import annotations
import hashlib
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from rdflib import Graph


_EMBED_DIM = 384


def _graph_hash(graph: "Graph") -> str:
    """Fast fingerprint: sorted first 500 subject URIs + triple count.

    Sort before slicing so the sample is stable across Python runs
    (set/dict iteration order is not guaranteed to be consistent).
    """
    subjects = sorted(str(s) for s in graph.subjects())[:500]
    fingerprint = "|".join(subjects) + f"|count={len(graph)}"
    return hashlib.md5(fingerprint.encode()).hexdigest()


class LanceStore:
    """Thin wrapper around a LanceDB directory for vector persistence."""

    def __init__(self, path: str) -> None:
        try:
            import lancedb
        except ImportError as exc:
            raise ImportError(
                "lancedb is required for persistent indexing: pip install lancedb"
            ) from exc
        self.path = path
        self._db = lancedb.connect(path)

    # ------------------------------------------------------------------
    # Staleness
    # ------------------------------------------------------------------

    def is_valid(self, graph: "Graph") -> bool:
        """Return True if the cached index matches the current graph."""
        try:
            meta = self._db.open_table("meta").to_pandas()
            if meta.empty:
                return False
            return (
                meta["graph_hash"].iloc[0] == _graph_hash(graph)
                and int(meta["triple_count"].iloc[0]) == len(graph)
            )
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self, rows: list[dict], graph: "Graph") -> None:
        """Persist rows and store graph fingerprint."""
        import pandas as pd

        if not rows:
            return

        df = pd.DataFrame(
            {
                "component": [r["component"] for r in rows],
                "term_key": [r["term_key"] for r in rows],
                "chunk_idx": [r["chunk_idx"] for r in rows],
                "vector": [r["vector"].tolist() for r in rows],
            }
        )
        self._db.create_table("embeddings", data=df, mode="overwrite")

        meta_df = pd.DataFrame(
            {
                "graph_hash": [_graph_hash(graph)],
                "triple_count": [len(graph)],
            }
        )
        self._db.create_table("meta", data=meta_df, mode="overwrite")

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_rows(self) -> list[dict]:
        """Load all embedding rows from disk — no re-embedding needed."""
        df = self._db.open_table("embeddings").to_pandas()
        rows = []
        for _, row in df.iterrows():
            rows.append(
                {
                    "component": row["component"],
                    "term_key": row["term_key"],
                    "chunk_idx": int(row["chunk_idx"]),
                    "vector": np.array(row["vector"], dtype=np.float32),
                }
            )
        return rows

    # ------------------------------------------------------------------
    # Incremental append
    # ------------------------------------------------------------------

    def append(self, rows: list[dict]) -> None:
        """Append new rows (for incremental triple additions)."""
        if not rows:
            return
        import pandas as pd

        df = pd.DataFrame(
            {
                "component": [r["component"] for r in rows],
                "term_key": [r["term_key"] for r in rows],
                "chunk_idx": [r["chunk_idx"] for r in rows],
                "vector": [r["vector"].tolist() for r in rows],
            }
        )
        try:
            self._db.open_table("embeddings").add(df)
        except Exception:
            self._db.create_table("embeddings", data=df, mode="overwrite")

    # ------------------------------------------------------------------
    # ANN search (used by top_k when index is available)
    # ------------------------------------------------------------------

    def ann_search(self, query_vec: np.ndarray, component: str, k: int) -> list[tuple[str, float]]:
        """Return top-k (term_key, score) pairs using HNSW ANN search."""
        tbl = self._db.open_table("embeddings")
        results = (
            tbl.search(query_vec.tolist())
            .where(f"component = '{component}'")
            .limit(k * 3)  # over-fetch to account for multiple chunks per term
            .select(["term_key", "_distance"])
            .to_pandas()
        )
        # Convert distance → cosine similarity (LanceDB returns L2 distance for float vectors)
        # For normalized vectors: cosine_sim = 1 - l2_dist² / 2
        seen: dict[str, float] = {}
        for _, row in results.iterrows():
            key = row["term_key"]
            # _distance is L2; for unit vectors cos_sim = 1 - dist²/2
            d = float(row["_distance"])
            score = max(0.0, 1.0 - d * d / 2.0)
            if score > seen.get(key, -1.0):
                seen[key] = score
        return sorted(seen.items(), key=lambda x: x[1], reverse=True)[:k]

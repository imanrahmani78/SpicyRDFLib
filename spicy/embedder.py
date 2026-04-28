from __future__ import annotations
import numpy as np

_model = None
_cache: dict[str, np.ndarray] = {}


def _get_model():
    global _model
    if _model is None:
        import torch
        from sentence_transformers import SentenceTransformer
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        _model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    return _model


def embed(text: str) -> np.ndarray:
    if text not in _cache:
        vec = _get_model().encode(text, normalize_embeddings=True)
        vec = np.array(vec, dtype=np.float32)
        vec.flags.writeable = False
        _cache[text] = vec
    return _cache[text]


def batch_embed(texts: list[str], batch_size: int = 256) -> list[np.ndarray]:
    """Encode a list of strings in one batched model call, populating the cache."""
    uncached = list({t for t in texts if t not in _cache})
    if uncached:
        model = _get_model()
        vecs = model.encode(
            uncached,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
        )
        for text, vec in zip(uncached, vecs):
            vec = np.array(vec, dtype=np.float32)
            vec.flags.writeable = False
            _cache[text] = vec
    return [_cache[t] for t in texts]

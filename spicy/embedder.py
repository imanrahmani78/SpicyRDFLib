from __future__ import annotations
import re
import numpy as np

_model = None
_cache: dict[str, np.ndarray] = {}

_SENT_RE = re.compile(r'(?<=[.!?])\s+|\n+')


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


def chunk_text(text: str, max_chars: int = 300) -> list[str]:
    """Split text into sentence-level chunks capped at max_chars.

    Short texts (≤ max_chars) are returned as-is.  Long texts are split on
    sentence boundaries first; sentences that are themselves longer than
    max_chars are hard-truncated as a last resort.
    """
    if len(text) <= max_chars:
        return [text]

    sentences = [s.strip() for s in _SENT_RE.split(text) if s.strip()]
    chunks: list[str] = []
    current = ""
    for sent in sentences:
        if len(sent) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            # hard truncate oversized single sentence
            for start in range(0, len(sent), max_chars):
                chunks.append(sent[start : start + max_chars])
        elif current and len(current) + 1 + len(sent) > max_chars:
            chunks.append(current)
            current = sent
        else:
            current = (current + " " + sent).strip() if current else sent
    if current:
        chunks.append(current)
    return chunks or [text[:max_chars]]


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

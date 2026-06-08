"""
Singleton sentence-transformer embedder.

Uses all-MiniLM-L6-v2 (80 MB, 384-dim).  Lazy-loaded on first call so
importing this module never crashes if sentence-transformers isn't installed
yet — it only fails at the moment you actually ask for an embedding.

Both MongoDB Atlas Vector Search and Elasticsearch dense_vector fields
use the same 384-dim vectors so the two services are always in sync.
"""

from __future__ import annotations
import threading
from typing import Optional


class _Embedder:
    _instance: Optional["_Embedder"] = None
    _lock = threading.Lock()

    MODEL_NAME = "all-MiniLM-L6-v2"
    DIM        = 384

    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for vector search.\n"
                "Install it: pip install sentence-transformers"
            ) from exc
        self._model = SentenceTransformer(self.MODEL_NAME)

    @classmethod
    def get(cls) -> "_Embedder":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def embed(self, text: str) -> list[float]:
        """Embed a single string → list[float] of length DIM."""
        vec = self._model.encode([text], convert_to_numpy=True)[0]
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple strings in one forward pass."""
        if not texts:
            return []
        vecs = self._model.encode(texts, convert_to_numpy=True, batch_size=32)
        return [v.tolist() for v in vecs]


def embed(text: str) -> list[float]:
    """Convenience wrapper — embed a single description string."""
    return _Embedder.get().embed(text)


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Convenience wrapper — embed a list of strings."""
    return _Embedder.get().embed_batch(texts)


DIM = _Embedder.DIM

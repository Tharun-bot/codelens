"""FAISS vector index: store embeddings, run nearest-neighbor search, persist to disk.

Uses IndexFlatIP (exact search via inner product). Vectors coming from
Embedder are already L2-normalized, so inner product == cosine similarity.
Flat index does brute-force exact search — fine up to ~500k vectors, which
covers the vast majority of real codebases. For 10M+ line codebases you'd
swap this for an IVF (approximate) index instead.
"""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np


class VectorIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self._index = faiss.IndexFlatIP(dim)

    @property
    def ntotal(self) -> int:
        """Number of vectors currently stored in the index."""
        return self._index.ntotal

    def build(self, vectors: np.ndarray) -> None:
        """
        Reset the index and add `vectors` as its initial contents.
        vectors: shape (N, dim), float32, ideally L2-normalized already.
        """
        self._validate(vectors)
        self._index = faiss.IndexFlatIP(self.dim)
        self._index.add(vectors)

    def add(self, vectors: np.ndarray) -> None:
        """Append more vectors to an existing index (does not reset)."""
        self._validate(vectors)
        self._index.add(vectors)

    def search(self, query_vector: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """
        Search for the k nearest vectors to `query_vector`.

        Args:
            query_vector: shape (dim,) or (1, dim), float32
            k: number of results to return

        Returns:
            (scores, ids): each shape (k,). scores are similarity scores
            (higher = more similar). ids are row indices into the vectors
            you originally passed to build()/add() — use these to look up
            metadata (file path, line numbers, etc.) elsewhere.
        """
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        k = min(k, self._index.ntotal) if self._index.ntotal > 0 else 0
        if k == 0:
            return np.array([]), np.array([])

        scores, ids = self._index.search(query_vector.astype("float32"), k)
        return scores[0], ids[0]

    def save(self, path: str | Path) -> None:
        """Persist the index to disk."""
        faiss.write_index(self._index, str(path))

    def load(self, path: str | Path) -> None:
        """Load a previously saved index from disk, replacing current contents."""
        self._index = faiss.read_index(str(path))
        self.dim = self._index.d

    def _validate(self, vectors: np.ndarray) -> None:
        if vectors.ndim != 2:
            raise ValueError(f"Expected 2D array (N, dim), got shape {vectors.shape}")
        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"Vector dim mismatch: index expects dim={self.dim}, got {vectors.shape[1]}"
            )
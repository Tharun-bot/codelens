"""Embedding generation for code chunks and search queries.

Uses sentence-transformers. Default model is all-MiniLM-L6-v2 (general
purpose, fast, 384-dim). This is swappable — a code-specific model like
microsoft/codebert-base or a CodeSearchNet-trained model can be dropped in
via the `model_name` param without changing any calling code.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from codelens.chunker import CodeChunk

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    @property
    def embedding_dim(self) -> int:
        return self._model.get_embedding_dimension()

    def embed_chunks(self, chunks: list[CodeChunk]) -> np.ndarray:
        """
        Embed a list of CodeChunks. Returns an (N, dim) float32 array,
        L2-normalized so inner product == cosine similarity.

        We embed name + docstring + source_text concatenated together
        (not just raw code) — this improves matching against natural
        language queries, since the docstring/name carry intent in plain
        English while the body is pure code.
        """
        texts = [self._chunk_to_text(c) for c in chunks]
        return self._embed_texts(texts)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single natural language query string. Returns shape (dim,)."""
        vec = self._embed_texts([text])
        return vec[0]

    def _chunk_to_text(self, chunk: CodeChunk) -> str:
        parts = [chunk.name]
        if chunk.docstring:
            parts.append(chunk.docstring)
        parts.append(chunk.source_text)
        return "\n".join(parts)

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,  # so cosine similarity = dot product
            show_progress_bar=False,
        )
        return embeddings.astype("float32")
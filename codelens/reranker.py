"""Cross-encoder re-ranking.

The bi-encoder (Phase 3) embeds query and chunks independently, then FAISS
finds nearest neighbors by vector distance — fast, but less precise, since
the query and chunk never actually "see" each other during scoring.

A cross-encoder takes the (query, chunk) pair together as joint input and
outputs a single relevance score — much more accurate, but too slow to run
against every chunk in a codebase. So the pattern is: bi-encoder retrieves a
shortlist (top-10 via FAISS), cross-encoder re-scores just that shortlist.
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from codelens.db import Chunk

DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self, model_name: str = DEFAULT_CROSS_ENCODER_MODEL):
        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[Chunk]) -> list[tuple[Chunk, float]]:
        """
        Re-score and re-sort candidate chunks against the query.

        Args:
            query: natural language search query
            candidates: chunks retrieved by the bi-encoder/FAISS stage

        Returns:
            List of (Chunk, score) tuples, sorted by cross-encoder score
            descending. Scores are raw cross-encoder logits — not bounded
            like cosine similarity, only meaningful in relative order.
        """
        if not candidates:
            return []

        pairs = [(query, self._chunk_to_text(c)) for c in candidates]
        scores = self._model.predict(pairs)

        scored = list(zip(candidates, scores))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [(chunk, float(score)) for chunk, score in scored]

    def _chunk_to_text(self, chunk: Chunk) -> str:
        parts = [chunk.name]
        if chunk.docstring:
            parts.append(chunk.docstring)
        parts.append(chunk.source_text)
        return "\n".join(parts)
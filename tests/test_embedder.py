"""Tests for codelens.embedder — sentence-transformers embedding pipeline."""

import numpy as np
import pytest

from codelens.chunker import CodeChunk
from codelens.embedder import Embedder

# Model load is slow (~few seconds + first-run download) — share one instance
# across all tests in this file instead of reloading per test.
@pytest.fixture(scope="module")
def embedder() -> Embedder:
    return Embedder()


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # vectors are already normalized by the embedder, so dot product = cosine sim
    return float(np.dot(a, b))


@pytest.mark.slow
def test_embedding_shape(embedder: Embedder):
    vec = embedder.embed_query("find the login function")
    assert vec.shape == (384,)
    assert vec.dtype == np.float32


@pytest.mark.slow
def test_similar_code_has_high_similarity(embedder: Embedder):
    chunk_a = CodeChunk(
        file_path="a.py", name="add", node_type="function",
        start_line=1, end_line=2,
        source_text="def add(a, b):\n    return a + b",
    )
    chunk_b = CodeChunk(
        file_path="b.py", name="add_numbers", node_type="function",
        start_line=1, end_line=2,
        source_text="def add_numbers(x, y):\n    return x + y",
    )

    vecs = embedder.embed_chunks([chunk_a, chunk_b])
    sim = _cosine_sim(vecs[0], vecs[1])

    assert sim > 0.7


@pytest.mark.slow
def test_unrelated_code_has_lower_similarity_than_similar_code(embedder: Embedder):
    chunk_add = CodeChunk(
        file_path="a.py", name="add", node_type="function",
        start_line=1, end_line=2,
        source_text="def add(a, b):\n    return a + b",
    )
    chunk_add_variant = CodeChunk(
        file_path="b.py", name="sum_two", node_type="function",
        start_line=1, end_line=2,
        source_text="def sum_two(x, y):\n    return x + y",
    )
    chunk_unrelated = CodeChunk(
        file_path="c.py", name="send_email", node_type="function",
        start_line=1, end_line=3,
        source_text="def send_email(to, subject, body):\n    smtp.send(to, subject, body)",
    )

    vecs = embedder.embed_chunks([chunk_add, chunk_add_variant, chunk_unrelated])
    sim_related = _cosine_sim(vecs[0], vecs[1])
    sim_unrelated = _cosine_sim(vecs[0], vecs[2])

    assert sim_related > sim_unrelated


@pytest.mark.slow
def test_query_and_matching_code_are_reasonably_close(embedder: Embedder):
    chunk = CodeChunk(
        file_path="auth.py", name="refresh_token", node_type="function",
        start_line=1, end_line=4,
        docstring="Refresh an expired auth token using the refresh token.",
        source_text="def refresh_token(token):\n    validate(token)\n    return issue_new(token)",
    )
    query_vec = embedder.embed_query("how do I refresh an auth token")
    chunk_vec = embedder.embed_chunks([chunk])[0]

    sim = _cosine_sim(query_vec, chunk_vec)
    # not a strict bar — just proving the NL query lands in the right neighborhood,
    # well above what a random/unrelated pairing would score
    assert sim > 0.3


@pytest.mark.slow
def test_embedder_exposes_embedding_dim(embedder: Embedder):
    assert embedder.embedding_dim == 384
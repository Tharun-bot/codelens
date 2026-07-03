"""Tests for codelens.vector_index — FAISS build, search, and persistence."""

from pathlib import Path

import numpy as np
import pytest

from codelens.vector_index import VectorIndex

DIM = 16


def _normalized_random_vectors(n: int, dim: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vecs = rng.random((n, dim)).astype("float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


@pytest.fixture
def sample_vectors() -> np.ndarray:
    return _normalized_random_vectors(5, DIM)


def test_build_sets_ntotal(sample_vectors):
    index = VectorIndex(dim=DIM)
    index.build(sample_vectors)
    assert index.ntotal == 5


def test_search_finds_exact_match(sample_vectors):
    index = VectorIndex(dim=DIM)
    index.build(sample_vectors)

    query = sample_vectors[2]
    scores, ids = index.search(query, k=3)

    assert ids[0] == 2
    assert scores[0] > 0.999  # searching for itself -> near-perfect similarity


def test_search_returns_k_results_in_descending_score_order(sample_vectors):
    index = VectorIndex(dim=DIM)
    index.build(sample_vectors)

    scores, ids = index.search(sample_vectors[0], k=3)

    assert len(ids) == 3
    assert list(scores) == sorted(scores, reverse=True)


def test_search_caps_k_at_ntotal(sample_vectors):
    index = VectorIndex(dim=DIM)
    index.build(sample_vectors)

    # only 5 vectors exist, ask for 100
    scores, ids = index.search(sample_vectors[0], k=100)
    assert len(ids) == 5


def test_add_appends_without_resetting(sample_vectors):
    index = VectorIndex(dim=DIM)
    index.build(sample_vectors[:3])
    assert index.ntotal == 3

    index.add(sample_vectors[3:])
    assert index.ntotal == 5


def test_dim_mismatch_raises():
    index = VectorIndex(dim=DIM)
    wrong_dim_vectors = _normalized_random_vectors(3, DIM + 1)

    with pytest.raises(ValueError):
        index.build(wrong_dim_vectors)


def test_save_and_load_roundtrip(sample_vectors, tmp_path: Path):
    index = VectorIndex(dim=DIM)
    index.build(sample_vectors)

    save_path = tmp_path / "test.faiss"
    index.save(save_path)

    loaded_index = VectorIndex(dim=DIM)
    loaded_index.load(save_path)

    assert loaded_index.ntotal == index.ntotal

    query = sample_vectors[2]
    orig_scores, orig_ids = index.search(query, k=3)
    loaded_scores, loaded_ids = loaded_index.search(query, k=3)

    assert list(orig_ids) == list(loaded_ids)
    np.testing.assert_allclose(orig_scores, loaded_scores, rtol=1e-5)


def test_search_on_empty_index_returns_empty():
    index = VectorIndex(dim=DIM)
    query = _normalized_random_vectors(1, DIM)[0]

    scores, ids = index.search(query, k=5)
    assert len(scores) == 0
    assert len(ids) == 0
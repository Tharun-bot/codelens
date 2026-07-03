"""Tests for codelens.reranker — cross-encoder re-ranking."""

import pytest

from codelens.db import Chunk
from codelens.reranker import Reranker


def _make_chunk(name: str, source_text: str, docstring: str | None = None) -> Chunk:
    # not persisted to a DB — just building an in-memory object with the
    # fields Reranker actually reads
    return Chunk(
        id=None,
        repo_id=1,
        file_path=f"{name}.py",
        node_type="function",
        name=name,
        start_line=1,
        end_line=2,
        source_text=source_text,
        docstring=docstring,
        faiss_id=0,
    )


@pytest.fixture(scope="module")
def reranker() -> Reranker:
    return Reranker()


@pytest.mark.slow
def test_rerank_promotes_most_relevant_candidate(reranker: Reranker):
    query = "parse JSON config file"

    json_chunk = _make_chunk(
        "load_config",
        "def load_config(path):\n    with open(path) as f:\n        return json.load(f)",
        docstring="Parse a JSON configuration file from disk.",
    )
    http_chunk = _make_chunk(
        "fetch_url",
        "def fetch_url(url):\n    return requests.get(url).text",
    )
    string_chunk = _make_chunk(
        "reverse_string",
        "def reverse_string(s):\n    return s[::-1]",
    )

    # deliberately shuffled so the relevant one isn't first going in
    candidates = [string_chunk, http_chunk, json_chunk]

    reranked = reranker.rerank(query, candidates)

    assert reranked[0][0].name == "load_config"


@pytest.mark.slow
def test_rerank_returns_scores_in_descending_order(reranker: Reranker):
    query = "send an email notification"

    chunks = [
        _make_chunk("reverse_string", "def reverse_string(s):\n    return s[::-1]"),
        _make_chunk("send_email", "def send_email(to, subject):\n    smtp.send(to, subject)"),
    ]

    reranked = reranker.rerank(query, chunks)
    scores = [score for _, score in reranked]

    assert scores == sorted(scores, reverse=True)


@pytest.mark.slow
def test_rerank_preserves_all_candidates(reranker: Reranker):
    query = "anything"
    chunks = [
        _make_chunk("a", "def a(): pass"),
        _make_chunk("b", "def b(): pass"),
        _make_chunk("c", "def c(): pass"),
    ]

    reranked = reranker.rerank(query, chunks)
    assert len(reranked) == 3
    assert {c.name for c, _ in reranked} == {"a", "b", "c"}


def test_rerank_empty_candidates_returns_empty():
    # doesn't need the model at all — should short-circuit
    reranker = Reranker.__new__(Reranker)  # skip __init__, avoid loading model
    result = reranker.rerank("query", [])
    assert result == []
"""Tests for codelens.pipeline — end-to-end indexing."""

from pathlib import Path

import pytest

from codelens.db import get_chunks_by_faiss_ids, get_engine, get_session, init_db
from codelens.embedder import Embedder
from codelens.pipeline import index_repository
from codelens.vector_index import VectorIndex

TEST_DATABASE_URL = "sqlite:///:memory:"


@pytest.fixture
def fixture_repo(tmp_path: Path) -> Path:
    """A tiny repo: 2 files, 3 functions total."""
    (tmp_path / "math_utils.py").write_text(
        "def add(a, b):\n"
        "    return a + b\n"
        "\n"
        "def subtract(a, b):\n"
        "    return a - b\n"
    )
    (tmp_path / "auth.py").write_text(
        "def refresh_token(token):\n"
        "    \"\"\"Refresh an expired auth token.\"\"\"\n"
        "    return issue_new(token)\n"
    )
    return tmp_path


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    # shared across tests in this file to avoid reloading the model repeatedly
    return Embedder()


@pytest.mark.slow
def test_index_repository_returns_correct_chunk_count(fixture_repo, embedder, tmp_path):
    result = index_repository(
        fixture_repo,
        index_dir=tmp_path / "indexes",
        embedder=embedder,
        database_url=TEST_DATABASE_URL,
    )
    assert result.chunks_indexed == 3


@pytest.mark.slow
def test_faiss_index_matches_chunk_count(fixture_repo, embedder, tmp_path):
    result = index_repository(
        fixture_repo,
        index_dir=tmp_path / "indexes",
        embedder=embedder,
        database_url=TEST_DATABASE_URL,
    )

    vector_index = VectorIndex(dim=embedder.embedding_dim)
    vector_index.load(result.index_path)

    assert vector_index.ntotal == result.chunks_indexed


@pytest.mark.slow
def test_faiss_id_correctly_maps_to_postgres_row(fixture_repo, embedder, tmp_path):
    result = index_repository(
        fixture_repo,
        index_dir=tmp_path / "indexes",
        embedder=embedder,
        database_url=TEST_DATABASE_URL,
    )

    # NOTE: this test relies on SQLite persisting for the process lifetime.
    # In-memory SQLite is per-connection, so instead we verify structurally:
    # re-run indexing against a *file-based* SQLite so we can reconnect and check.
    db_path = tmp_path / "test.db"
    file_db_url = f"sqlite:///{db_path}"

    result2 = index_repository(
        fixture_repo,
        index_dir=tmp_path / "indexes2",
        embedder=embedder,
        database_url=file_db_url,
    )

    engine = get_engine(database_url=file_db_url)
    init_db(engine)
    session = get_session(engine)
    try:
        chunks = get_chunks_by_faiss_ids(session, result2.repo_id, [0, 1, 2])
        assert len(chunks) == 3
        names = {c.name for c in chunks}
        assert names == {"add", "subtract", "refresh_token"}
    finally:
        session.close()


@pytest.mark.slow
def test_index_repository_raises_on_empty_repo(tmp_path, embedder):
    empty_dir = tmp_path / "empty_repo"
    empty_dir.mkdir()

    with pytest.raises(ValueError):
        index_repository(
            empty_dir,
            index_dir=tmp_path / "indexes",
            embedder=embedder,
            database_url=TEST_DATABASE_URL,
        )


@pytest.mark.slow
def test_index_file_is_actually_written_to_disk(fixture_repo, embedder, tmp_path):
    index_dir = tmp_path / "indexes"
    result = index_repository(
        fixture_repo,
        index_dir=index_dir,
        embedder=embedder,
        database_url=TEST_DATABASE_URL,
    )

    assert Path(result.index_path).exists()
    assert Path(result.index_path).parent == index_dir
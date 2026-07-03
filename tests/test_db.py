"""Tests for codelens.db — metadata persistence.

Runs against an in-memory SQLite engine (via get_engine() with no
DATABASE_URL set) so these tests don't require a real Supabase connection.
The same code path (SQLAlchemy ORM, no Postgres-specific SQL) runs
identically against real Postgres in production.
"""

import pytest

from codelens.db import (
    create_repo,
    get_chunks_by_faiss_ids,
    get_engine,
    get_session,
    init_db,
    insert_chunks,
)


@pytest.fixture
def session():
    engine = get_engine(database_url="sqlite:///:memory:")
    init_db(engine)
    s = get_session(engine)
    yield s
    s.close()


def _sample_chunk_dict(faiss_id: int, name: str = "add") -> dict:
    return {
        "file_path": "a.py",
        "node_type": "function",
        "name": name,
        "start_line": 1,
        "end_line": 2,
        "source_text": f"def {name}(a, b):\n    return a + b",
        "docstring": None,
        "faiss_id": faiss_id,
    }


def test_create_repo(session):
    repo = create_repo(session, "/some/local/path")
    assert repo.id is not None
    assert repo.path_or_url == "/some/local/path"
    assert repo.indexed_at is not None


def test_insert_and_fetch_chunks_roundtrip(session):
    repo = create_repo(session, "/some/repo")
    chunk_dicts = [_sample_chunk_dict(faiss_id=0, name="add"), _sample_chunk_dict(faiss_id=1, name="subtract")]

    inserted = insert_chunks(session, repo.id, chunk_dicts)
    assert len(inserted) == 2

    fetched = get_chunks_by_faiss_ids(session, repo.id, [0, 1])
    names = {c.name for c in fetched}
    assert names == {"add", "subtract"}


def test_fetch_by_faiss_id_returns_correct_fields(session):
    repo = create_repo(session, "/some/repo")
    insert_chunks(session, repo.id, [_sample_chunk_dict(faiss_id=5, name="multiply")])

    fetched = get_chunks_by_faiss_ids(session, repo.id, [5])
    assert len(fetched) == 1
    chunk = fetched[0]
    assert chunk.name == "multiply"
    assert chunk.file_path == "a.py"
    assert chunk.start_line == 1
    assert chunk.end_line == 2


def test_fetch_nonexistent_faiss_id_returns_empty(session):
    repo = create_repo(session, "/some/repo")
    insert_chunks(session, repo.id, [_sample_chunk_dict(faiss_id=0)])

    fetched = get_chunks_by_faiss_ids(session, repo.id, [999])
    assert fetched == []


def test_fetch_scoped_to_repo(session):
    repo_a = create_repo(session, "/repo/a")
    repo_b = create_repo(session, "/repo/b")

    # both repos happen to reuse faiss_id=0 (valid — faiss_id is per-index, not globally unique)
    insert_chunks(session, repo_a.id, [_sample_chunk_dict(faiss_id=0, name="from_a")])
    insert_chunks(session, repo_b.id, [_sample_chunk_dict(faiss_id=0, name="from_b")])

    fetched_a = get_chunks_by_faiss_ids(session, repo_a.id, [0])
    assert len(fetched_a) == 1
    assert fetched_a[0].name == "from_a"


def test_get_engine_defaults_to_sqlite_when_no_url_set(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    engine = get_engine(database_url="sqlite:///:memory:")
    assert "sqlite" in str(engine.url)
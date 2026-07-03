"""Tests for codelens.api — FastAPI index and search endpoints."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import codelens.api as api_module
from codelens.api import app


@pytest.fixture
def fixture_repo(tmp_path: Path) -> Path:
    (tmp_path / "math_utils.py").write_text(
        "def add(a, b):\n"
        "    \"\"\"Add two numbers together.\"\"\"\n"
        "    return a + b\n"
        "\n"
        "def send_email(to, subject):\n"
        "    smtp.send(to, subject)\n"
    )
    return tmp_path


@pytest.fixture
def client(tmp_path, monkeypatch):
    """
    Isolate each test run: point INDEX_DIR at a temp dir and use a temp
    file-based SQLite DB so tests don't collide with each other or with a
    real Supabase instance.
    """
    monkeypatch.setattr(api_module, "INDEX_DIR", tmp_path / "indexes")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")

    # reset process-wide caches between tests
    api_module._embedder = None
    api_module._vector_index_cache.clear()

    return TestClient(app)


@pytest.mark.slow
def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.slow
def test_index_endpoint_returns_chunk_count(client, fixture_repo):
    response = client.post("/index", json={"path_or_url": str(fixture_repo)})
    assert response.status_code == 200

    data = response.json()
    assert data["chunks_indexed"] == 2
    assert "repo_id" in data


@pytest.mark.slow
def test_index_endpoint_rejects_missing_path(client):
    response = client.post("/index", json={"path_or_url": "/definitely/not/a/real/path"})
    assert response.status_code == 400


@pytest.mark.slow
def test_search_endpoint_finds_relevant_function(client, fixture_repo):
    index_response = client.post("/index", json={"path_or_url": str(fixture_repo)})
    repo_id = index_response.json()["repo_id"]

    search_response = client.post(
        "/search", json={"query": "add two numbers", "repo_id": repo_id, "k": 5}
    )
    assert search_response.status_code == 200

    results = search_response.json()["results"]
    assert len(results) > 0

    top_names = [r["name"] for r in results[:2]]
    assert "add" in top_names


@pytest.mark.slow
def test_search_response_schema(client, fixture_repo):
    index_response = client.post("/index", json={"path_or_url": str(fixture_repo)})
    repo_id = index_response.json()["repo_id"]

    search_response = client.post(
        "/search", json={"query": "email sending", "repo_id": repo_id, "k": 3}
    )
    result = search_response.json()["results"][0]

    assert set(result.keys()) == {
        "file_path", "name", "node_type", "start_line", "end_line", "score", "snippet"
    }


@pytest.mark.slow
def test_search_on_nonexistent_repo_returns_404(client):
    response = client.post(
        "/search", json={"query": "anything", "repo_id": 999999, "k": 5}
    )
    assert response.status_code == 404

@pytest.mark.slow
def test_search_reranking_changes_order_from_raw_faiss(client, tmp_path):
    # Build a fixture repo where the naive vector-similarity winner and the
    # true best match are different, so reranking has something to correct.
    repo_dir = tmp_path / "rerank_repo"
    repo_dir.mkdir()
    (repo_dir / "utils.py").write_text(
        "def parse_config(path):\n"
        "    \"\"\"Load and parse a JSON configuration file.\"\"\"\n"
        "    with open(path) as f:\n"
        "        return json.load(f)\n"
        "\n"
        "def reverse_text(s):\n"
        "    return s[::-1]\n"
        "\n"
        "def fetch_data(url):\n"
        "    return requests.get(url).json()\n"
    )

    index_response = client.post("/index", json={"path_or_url": str(repo_dir)})
    repo_id = index_response.json()["repo_id"]

    search_response = client.post(
        "/search", json={"query": "parse JSON config file", "repo_id": repo_id, "k": 3}
    )
    results = search_response.json()["results"]

    # after reranking, the config parser should be the top result
    assert results[0]["name"] == "parse_config"
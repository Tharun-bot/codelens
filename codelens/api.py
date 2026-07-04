"""FastAPI service exposing index and search endpoints.

This wraps the pipeline (Phase 6) and vector search (Phase 4) behind a REST
API. The embedder and any loaded VectorIndex instances are cached in memory
per-process so we don't reload the model or re-read the index from disk on
every request.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from codelens.db import get_chunks_by_faiss_ids, get_engine, get_session, init_db
from codelens.embedder import Embedder
from codelens.pipeline import index_repository
from codelens.vector_index import VectorIndex
from codelens.reranker import Reranker
from codelens.ingest import is_git_url
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="CodeLens", description="Semantic code search engine")

# Where FAISS index files live on disk
INDEX_DIR = Path("data/indexes")

# Process-wide caches — avoid reloading the embedding model or re-reading
# FAISS index files from disk on every single request.
_embedder: Embedder | None = None
_vector_index_cache: dict[int, VectorIndex] = {}
_reranker: Reranker | None = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder

def get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker


def get_vector_index(repo_id: int) -> VectorIndex:
    if repo_id not in _vector_index_cache:
        index_path = INDEX_DIR / f"{repo_id}.faiss"
        if not index_path.exists():
            raise HTTPException(status_code=404, detail=f"No index found for repo_id {repo_id}")
        vi = VectorIndex(dim=get_embedder().embedding_dim)
        vi.load(index_path)
        _vector_index_cache[repo_id] = vi
    return _vector_index_cache[repo_id]


class IndexRequest(BaseModel):
    path_or_url: str


class IndexResponse(BaseModel):
    repo_id: int
    chunks_indexed: int


class SearchRequest(BaseModel):
    query: str
    repo_id: int
    k: int = 10


class SearchResult(BaseModel):
    file_path: str
    name: str
    node_type: str
    start_line: int
    end_line: int
    score: float
    snippet: str


class SearchResponse(BaseModel):
    results: list[SearchResult]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/index", response_model=IndexResponse)
def index_endpoint(request: IndexRequest):
    path_or_url = request.path_or_url
    if not is_git_url(path_or_url) and not Path(path_or_url).exists():
        raise HTTPException(status_code=400, detail=f"Path does not exist: {path_or_url}")
    try:
        result = index_repository(path_or_url, index_dir=INDEX_DIR, embedder=get_embedder())
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    # invalidate any stale cached index for this repo_id (shouldn't normally collide, but be safe)
    _vector_index_cache.pop(result.repo_id, None)

    return IndexResponse(repo_id=result.repo_id, chunks_indexed=result.chunks_indexed)


@app.post("/search", response_model=SearchResponse)
def search_endpoint(request: SearchRequest):
    vector_index = get_vector_index(request.repo_id)
    embedder = get_embedder()

    # Stage 1: bi-encoder retrieval via FAISS — cast a wide net (up to 10
    # candidates) even if the caller wants fewer final results, so the
    # cross-encoder has enough to meaningfully re-rank.
    query_vec = embedder.embed_query(request.query)
    retrieval_k = max(request.k, 10)
    scores, ids = vector_index.search(query_vec, k=retrieval_k)

    if len(ids) == 0:
        return SearchResponse(results=[])

    engine = get_engine()
    init_db(engine)
    session = get_session(engine)
    try:
        chunks = get_chunks_by_faiss_ids(session, request.repo_id, [int(i) for i in ids])
        chunks_by_faiss_id = {c.faiss_id: c for c in chunks}

        # preserve FAISS ranking order for the candidates we actually found metadata for
        candidates = [
            chunks_by_faiss_id[int(faiss_id)]
            for faiss_id in ids
            if int(faiss_id) in chunks_by_faiss_id
        ]

        # Stage 2: cross-encoder re-ranking on the shortlist
        reranker = get_reranker()
        reranked = reranker.rerank(request.query, candidates)

        results = [
            SearchResult(
                file_path=chunk.file_path,
                name=chunk.name,
                node_type=chunk.node_type,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                score=score,
                snippet=chunk.source_text,
            )
            for chunk, score in reranked[: request.k]
        ]
        return SearchResponse(results=results)
    finally:
        session.close()
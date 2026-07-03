"""End-to-end indexing pipeline: repo path -> discover -> chunk -> embed -> store.

This is the glue that turns everything built in Phases 1-5 into a single
callable: point it at a directory, get back a fully searchable index
(FAISS vectors on disk + metadata rows in Postgres/Supabase).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from codelens.chunker import CodeChunk, chunk_file
from codelens.db import create_repo, get_engine, get_session, init_db, insert_chunks
from codelens.embedder import Embedder
from codelens.ingest import discover_files, resolve_repo_source
from codelens.vector_index import VectorIndex


@dataclass
class IndexResult:
    repo_id: int
    chunks_indexed: int
    index_path: str


def index_repository(
    path: Path,
    index_dir: Path,
    embedder: Embedder | None = None,
    database_url: str | None = None,
) -> IndexResult:
    """
    Index a local repository end-to-end.

    Args:
        path: directory to index
        index_dir: directory to write the FAISS index file into
                   (file will be named "<repo_id>.faiss" once repo_id is known)
        embedder: optional pre-built Embedder (reused across calls to avoid
                   reloading the model each time); a fresh one is created if
                   not provided
        database_url: optional override; defaults to DATABASE_URL from .env,
                       falls back to in-memory SQLite if unset

    Returns:
        IndexResult with repo_id, chunk count, and the saved index file path.

    Raises:
        ValueError if no chunkable files are found under `path`.
    """
# Resolve to a local path — clones it first if `path` is actually a git URL
    resolved_path = resolve_repo_source(str(path))

    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    if embedder is None:
        embedder = Embedder()

    # 1. Discover files
    files = discover_files(resolved_path)

    # 2. Chunk each file
    all_chunks: list[CodeChunk] = []
    for file_path in files:
        all_chunks.extend(chunk_file(file_path))

    if not all_chunks:
        raise ValueError(f"No indexable code found under {path}")

    # 3. Embed all chunks
    vectors = embedder.embed_chunks(all_chunks)

    # 4. Build + save FAISS index
    vector_index = VectorIndex(dim=embedder.embedding_dim)
    vector_index.build(vectors)

    # 5. Persist metadata to Postgres/Supabase (or SQLite fallback)
    engine = get_engine(database_url=database_url)
    init_db(engine)
    session = get_session(engine)

    try:
        repo = create_repo(session, str(path))  # store original path/URL, not the temp clone dir

        chunk_dicts = [
            {
                "file_path": chunk.file_path,
                "node_type": chunk.node_type,
                "name": chunk.name,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "source_text": chunk.source_text,
                "docstring": chunk.docstring,
                "faiss_id": faiss_id,  # row index == position in `all_chunks` / `vectors`
            }
            for faiss_id, chunk in enumerate(all_chunks)
        ]
        insert_chunks(session, repo.id, chunk_dicts)

        index_path = index_dir / f"{repo.id}.faiss"
        vector_index.save(index_path)

        return IndexResult(
            repo_id=repo.id,
            chunks_indexed=len(all_chunks),
            index_path=str(index_path),
        )
    finally:
        session.close()
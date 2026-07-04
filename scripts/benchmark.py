"""
Benchmark CodeLens indexing and query performance against a real repository.

Usage:
    python scripts/benchmark.py --repo https://github.com/pallets/flask
    python scripts/benchmark.py --repo ./some/local/repo --queries 50

Writes a Markdown report to benchmarks/results.md.
"""

from __future__ import annotations

import argparse
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from codelens.embedder import Embedder
from codelens.pipeline import index_repository
from codelens.reranker import Reranker
from codelens.db import get_chunks_by_faiss_ids, get_engine, get_session, init_db
from codelens.vector_index import VectorIndex

# A varied set of natural-language queries to exercise search latency.
# Generic enough to return *something* on most real codebases.
DEFAULT_QUERIES = [
    "parse a configuration file",
    "handle an HTTP request",
    "validate user input",
    "connect to a database",
    "read a file from disk",
    "write data to a file",
    "send a network request",
    "log an error message",
    "authenticate a user",
    "serialize an object to JSON",
    "run tests",
    "create a class instance",
    "handle an exception",
    "loop over a list",
    "define a function decorator",
    "cache a computed value",
    "sort a collection",
    "format a string",
    "compute a hash",
    "close a connection",
]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(len(sorted_vals) * pct)
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx]


def run_benchmark(repo: str, num_queries: int, index_dir: Path) -> dict:
    embedder = Embedder()
    reranker = Reranker()

    # --- Indexing benchmark ---
    print(f"Indexing {repo} ...")
    start = time.perf_counter()
    result = index_repository(repo, index_dir=index_dir, embedder=embedder)
    index_time_s = time.perf_counter() - start
    print(f"Indexed {result.chunks_indexed} chunks in {index_time_s:.2f}s")

    # --- Query latency benchmark ---
    vector_index = VectorIndex(dim=embedder.embedding_dim)
    vector_index.load(Path(result.index_path))

    engine = get_engine()
    init_db(engine)
    session = get_session(engine)

    queries = (DEFAULT_QUERIES * ((num_queries // len(DEFAULT_QUERIES)) + 1))[:num_queries]

    latencies_ms = []
    try:
        for query in queries:
            start = time.perf_counter()

            query_vec = embedder.embed_query(query)
            scores, ids = vector_index.search(query_vec, k=10)

            if len(ids) > 0:
                chunks = get_chunks_by_faiss_ids(session, result.repo_id, [int(i) for i in ids])
                chunks_by_faiss_id = {c.faiss_id: c for c in chunks}
                candidates = [
                    chunks_by_faiss_id[int(faiss_id)]
                    for faiss_id in ids
                    if int(faiss_id) in chunks_by_faiss_id
                ]
                reranker.rerank(query, candidates)

            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies_ms.append(elapsed_ms)
    finally:
        session.close()

    return {
        "repo": repo,
        "chunks_indexed": result.chunks_indexed,
        "index_time_s": index_time_s,
        "num_queries": len(latencies_ms),
        "p50_ms": percentile(latencies_ms, 0.50),
        "p95_ms": percentile(latencies_ms, 0.95),
        "p99_ms": percentile(latencies_ms, 0.99),
        "mean_ms": statistics.mean(latencies_ms) if latencies_ms else 0.0,
        "min_ms": min(latencies_ms) if latencies_ms else 0.0,
        "max_ms": max(latencies_ms) if latencies_ms else 0.0,
    }


def write_report(stats: dict, output_path: Path) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    report = f"""# CodeLens Benchmark Results

Generated: {timestamp}

## Repository

- **Source:** `{stats['repo']}`
- **Chunks indexed:** {stats['chunks_indexed']}

## Indexing performance

- **Total index time:** {stats['index_time_s']:.2f}s

## Query latency (bi-encoder retrieval + cross-encoder rerank, end-to-end)

Measured over {stats['num_queries']} queries.

| Metric | Latency |
|---|---|
| p50 | {stats['p50_ms']:.1f} ms |
| p95 | {stats['p95_ms']:.1f} ms |
| p99 | {stats['p99_ms']:.1f} ms |
| mean | {stats['mean_ms']:.1f} ms |
| min | {stats['min_ms']:.1f} ms |
| max | {stats['max_ms']:.1f} ms |

## Notes

- Query latency includes: query embedding + FAISS search (k=10) + cross-encoder
  rerank on the top-10 candidates + Postgres/Supabase metadata lookup.
- FAISS search alone (bi-encoder retrieval only, before reranking) is
  sub-millisecond at this scale — most of the latency budget here is the
  cross-encoder pass and network round-trip to Supabase, not the vector search
  itself. This is the expected shape for the retrieve-then-rerank pattern.
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(f"\nReport written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark CodeLens indexing and search")
    parser.add_argument("--repo", required=True, help="Local path or git URL to benchmark against")
    parser.add_argument("--queries", type=int, default=20, help="Number of search queries to run")
    parser.add_argument("--index-dir", default="data/indexes", help="Directory for FAISS index files")
    parser.add_argument("--output", default="benchmarks/results.md", help="Path to write the report")
    args = parser.parse_args()

    stats = run_benchmark(args.repo, args.queries, Path(args.index_dir))
    write_report(stats, Path(args.output))

    print("\n--- Summary ---")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
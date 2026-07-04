# CodeLens Benchmark Results

Generated: 2026-07-04 09:55 UTC

## Repository

- **Source:** `https://github.com/pallets/flask`
- **Chunks indexed:** 1637

## Indexing performance

- **Total index time:** 313.02s

## Query latency (bi-encoder retrieval + cross-encoder rerank, end-to-end)

Measured over 30 queries.

| Metric | Latency |
|---|---|
| p50 | 290.8 ms |
| p95 | 583.9 ms |
| p99 | 683.3 ms |
| mean | 307.8 ms |
| min | 157.4 ms |
| max | 683.3 ms |

## Notes

- Query latency includes: query embedding + FAISS search (k=10) + cross-encoder
  rerank on the top-10 candidates + Postgres/Supabase metadata lookup.
- FAISS search alone (bi-encoder retrieval only, before reranking) is
  sub-millisecond at this scale — most of the latency budget here is the
  cross-encoder pass and network round-trip to Supabase, not the vector search
  itself. This is the expected shape for the retrieve-then-rerank pattern.

# CodeLens

**Semantic code search for developers** — search codebases by **meaning**, not exact text.

Instead of relying on `grep` or `Ctrl+F`, CodeLens understands the intent behind your query by indexing source code at the **function/class level** using AST-aware chunking and semantic embeddings.

> "Find the function that refreshes authentication tokens."
>
> "Where is request rate limiting implemented?"
>
> "Show me the code responsible for parsing configuration files."

---

## Overview

CodeLens is a semantic search engine for source code built around modern information retrieval techniques.

Each function or class is extracted using **tree-sitter**, converted into a vector embedding using **sentence-transformers**, stored in a **FAISS** vector index, and associated metadata is persisted in **Supabase (Postgres)**.

During search, CodeLens performs:

1. Semantic retrieval using a bi-encoder
2. Cross-encoder reranking
3. Metadata lookup to return the corresponding source code

This architecture provides significantly better results than keyword search while remaining fast enough for interactive use.

---

## Features

- Semantic search using natural language queries
- AST-aware chunking with tree-sitter
- Two-stage retrieval
  - Bi-encoder retrieval with FAISS
  - Cross-encoder reranking
- CLI and REST API
- Index local repositories or public GitHub repositories
- Supabase-backed metadata storage
- Docker support
- Benchmarking utilities

---

# Architecture

<img width="794" height="661" alt="image" src="https://github.com/user-attachments/assets/07ba5d3d-8a1e-4dba-ad6c-f5507c58d9e1" />


# Design Decisions

## Why FAISS?

FAISS is responsible solely for vector similarity search.

It performs extremely fast in-memory nearest-neighbor search and easily scales to hundreds of thousands of code chunks while maintaining sub-millisecond retrieval latency.

---

## Why Supabase?

Supabase (Postgres) stores all relational metadata, including:

- Repository information
- File paths
- Line numbers
- Source code
- Repository ownership

Separating vectors from metadata keeps the architecture simple, scalable, and maintainable.

---

## Why Two-Stage Retrieval?

CodeLens follows the same retrieve-then-rerank architecture used in many production search systems.

### Stage 1 — Bi-Encoder Retrieval

- Embed every code chunk once
- Embed the user query
- Retrieve the nearest neighbors using FAISS
- Extremely fast
- Produces a shortlist of candidates

### Stage 2 — Cross-Encoder Reranking

Each candidate is evaluated jointly with the query.

Although slower than the bi-encoder, the cross-encoder significantly improves ranking quality by considering the interaction between the query and the candidate code.

---

# Technology Stack

- Python
- FastAPI
- Typer
- tree-sitter
- sentence-transformers
- Cross-Encoder
- FAISS
- Supabase (Postgres)
- Docker

---

# Installation

## Clone the repository

```bash
git clone <repository-url>
cd codelens

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

---

## Configure Supabase

Create a `.env` file.

```bash
cp .env.example .env
```

Add your connection string.

```env
DATABASE_URL=postgresql://...
```

---

## Run the tests

```bash
pytest -v -m "not network"
```

---

# Usage

## CLI

### Index a local repository

```bash
codelens index ./my-project
```

### Index a GitHub repository

```bash
codelens index https://github.com/pallets/flask
```

### Search

```bash
codelens search \
    "parse a JSON config file" \
    --repo-id 1 \
    --k 5
```

---

## REST API

Start the server.

```bash
uvicorn codelens.api:app --reload
```

### Index a repository

```bash
curl -X POST http://localhost:8000/index \
-H "Content-Type: application/json" \
-d '{
  "path_or_url":"https://github.com/pallets/flask"
}'
```

### Search

```bash
curl -X POST http://localhost:8000/search \
-H "Content-Type: application/json" \
-d '{
  "query":"render a template",
  "repo_id":1,
  "k":5
}'
```

Interactive API documentation is available at:

```
http://localhost:8000/docs
```

---

# Docker

Build the image.

```bash
docker compose build
```

Start the services.

```bash
docker compose up -d
```

Health check.

```bash
curl http://localhost:8000/health
```

`DATABASE_URL` is automatically read from your `.env` file.

---

# Benchmarks

Repository used:

- Flask
- ~1,637 indexed code chunks
- CPU-only machine

## Indexing

| Metric | Value |
|---------|------:|
| Chunks Indexed | 1,637 |
| Total Index Time | 313 s |

---

## End-to-End Search Latency (30 Queries)

| Metric | Latency |
|---------|---------|
| p50 | 291 ms |
| p95 | 584 ms |
| p99 | 683 ms |

---

## Per-Stage Latency

| Stage | Latency |
|--------|--------:|
| Query Embedding | 31.1 ms |
| FAISS Search | 0.2 ms |
| Metadata Lookup | 262.9 ms |
| Cross-Encoder | 320.0 ms |
| Total | 614.2 ms |

The semantic retrieval step itself (FAISS) is not the bottleneck. The majority of end-to-end latency comes from:

- Cross-encoder inference
- Network round-trip to Supabase

This is expected in retrieve-then-rerank systems and is the tradeoff for improved ranking quality.

Future optimization opportunities include:

- GPU inference
- Smaller reranking models
- Regional database colocation
- Connection pooling

---

# Reproducing the Benchmark

```bash
python scripts/benchmark.py \
    --repo https://github.com/pallets/flask \
    --queries 30
```

---

# Project Structure

```text
codelens/
├── codelens/
│   ├── ingest.py
│   ├── chunker.py
│   ├── embedder.py
│   ├── vector_index.py
│   ├── db.py
│   ├── reranker.py
│   ├── pipeline.py
│   ├── api.py
│   └── cli.py
│
├── tests/
├── scripts/
│   └── benchmark.py
│
├── docs/
│   └── DESIGN.md
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

# Roadmap

- IVF-based FAISS indexes for larger repositories
- Incremental re-indexing
- Background indexing jobs
- Additional language grammars (Rust, Java, C++, Go)
- GPU inference for embedding and reranking
- Batch embedding during indexing
- Metadata caching
- Reduced database latency through regional deployment

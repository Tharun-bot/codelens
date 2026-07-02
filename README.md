# CodeLens

**Status: WIP (Phase 0 — scaffolding)**

A CLI + REST API tool that indexes a local codebase or GitHub repo and lets
you search it by *meaning*, not exact string match.

> "find the function that handles auth token refresh"
> "where is rate limiting implemented"

## Why

`ctrl+F` / `grep` search by exact string. Developers need semantic search over
code. This project builds that pipeline end-to-end: AST-aware chunking,
embeddings, vector search, and cross-encoder re-ranking.

## Stack

Python · FastAPI · sentence-transformers · FAISS · tree-sitter · PostgreSQL · Docker

## Architecture (evolving)
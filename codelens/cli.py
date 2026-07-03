"""CLI for CodeLens.

Usage:
    codelens index ./myrepo
    codelens search "jwt token validation" --repo ./myrepo

The CLI calls the pipeline and search logic directly (not over HTTP) — no
need for the FastAPI server to be running just to index/search locally.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from codelens.db import get_chunks_by_faiss_ids, get_engine, get_session, init_db
from codelens.embedder import Embedder
from codelens.pipeline import index_repository
from codelens.reranker import Reranker
from codelens.vector_index import VectorIndex

app = typer.Typer(help="CodeLens — semantic code search for your codebase.")
console = Console()

INDEX_DIR = Path("data/indexes")


@app.command()
def index(
    path: str = typer.Argument(..., help="Path to a local repository, or a git URL"),
):
    """Index a local repository or remote git repo so it can be searched."""
    from codelens.ingest import is_git_url

    if not is_git_url(path) and not Path(path).exists():
        console.print(f"[red]Path does not exist:[/red] {path}")
        raise typer.Exit(code=1)

    console.print(f"Indexing [bold]{path}[/bold]...")

    try:
        result = index_repository(path, index_dir=INDEX_DIR)
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)

    console.print(
        f"[green]Indexed {result.chunks_indexed} chunks[/green] "
        f"(repo_id={result.repo_id})"
    )


@app.command()
def search(
    query: str = typer.Argument(..., help="Natural language search query"),
    repo_id: int = typer.Option(..., "--repo-id", help="repo_id returned by the index command"),
    k: int = typer.Option(10, "--k", help="Number of results to return"),
):
    """Search a previously indexed repository."""
    index_path = INDEX_DIR / f"{repo_id}.faiss"
    if not index_path.exists():
        console.print(f"[red]No index found for repo_id {repo_id}. Run 'codelens index' first.[/red]")
        raise typer.Exit(code=1)

    embedder = Embedder()

    vector_index = VectorIndex(dim=embedder.embedding_dim)
    vector_index.load(index_path)

    query_vec = embedder.embed_query(query)
    retrieval_k = max(k, 10)
    scores, ids = vector_index.search(query_vec, k=retrieval_k)

    if len(ids) == 0:
        console.print("[yellow]No results found.[/yellow]")
        return

    engine = get_engine()
    init_db(engine)
    session = get_session(engine)
    try:
        chunks = get_chunks_by_faiss_ids(session, repo_id, [int(i) for i in ids])
        chunks_by_faiss_id = {c.faiss_id: c for c in chunks}
        candidates = [
            chunks_by_faiss_id[int(faiss_id)]
            for faiss_id in ids
            if int(faiss_id) in chunks_by_faiss_id
        ]

        reranker = Reranker()
        reranked = reranker.rerank(query, candidates)[:k]

        if not reranked:
            console.print("[yellow]No results found.[/yellow]")
            return

        table = Table(title=f'Results for "{query}"')
        table.add_column("Score", justify="right", style="cyan")
        table.add_column("Name", style="bold")
        table.add_column("Location", style="dim")

        for chunk, score in reranked:
            table.add_row(
                f"{score:.3f}",
                f"{chunk.name} ({chunk.node_type})",
                f"{chunk.file_path}:{chunk.start_line}-{chunk.end_line}",
            )

        console.print(table)

        # print snippets below the table
        for chunk, score in reranked:
            console.print(f"\n[bold]{chunk.file_path}:{chunk.start_line}-{chunk.end_line}[/bold]")
            console.print(chunk.source_text, style="dim")
    finally:
        session.close()


if __name__ == "__main__":
    app()
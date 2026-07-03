"""Metadata persistence layer.

FAISS stores vectors; Postgres (Supabase) stores everything about each
chunk that FAISS can't — file path, line numbers, source text, name — keyed
by the same integer id used as the vector's row index in FAISS. This lets us
go: query -> FAISS top-k ids -> Postgres lookup -> full result with metadata.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker


class Base(DeclarativeBase):
    pass


class Repo(Base):
    __tablename__ = "repos"

    id = Column(Integer, primary_key=True, autoincrement=True)
    path_or_url = Column(String, nullable=False)
    indexed_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    chunks = relationship("Chunk", back_populates="repo", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    repo_id = Column(Integer, ForeignKey("repos.id"), nullable=False)
    file_path = Column(String, nullable=False)
    node_type = Column(String, nullable=False)
    name = Column(String, nullable=False)
    start_line = Column(Integer, nullable=False)
    end_line = Column(Integer, nullable=False)
    source_text = Column(Text, nullable=False)
    docstring = Column(Text, nullable=True)
    faiss_id = Column(Integer, nullable=False)  # row index in the FAISS index

    repo = relationship("Repo", back_populates="chunks")


def get_engine(database_url: str | None = None):
    """
    Build a SQLAlchemy engine. If `database_url` isn't provided, reads
    DATABASE_URL from the environment (loaded from .env via python-dotenv).
    Falls back to an in-memory SQLite engine if neither is set — this is
    what lets tests run without any real database configured.
    """
    if database_url is None:
        from dotenv import load_dotenv

        load_dotenv()
        database_url = os.environ.get("DATABASE_URL")

    if database_url is None:
        database_url = "sqlite:///:memory:"

    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    return create_engine(database_url, connect_args=connect_args)


def init_db(engine) -> None:
    """Create all tables if they don't already exist."""
    Base.metadata.create_all(engine)


def get_session(engine) -> Session:
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


def create_repo(session: Session, path_or_url: str) -> Repo:
    repo = Repo(path_or_url=path_or_url)
    session.add(repo)
    session.commit()
    session.refresh(repo)
    return repo


def insert_chunks(session: Session, repo_id: int, chunks_with_faiss_ids: list[dict]) -> list[Chunk]:
    """
    Insert chunk metadata rows.

    Args:
        session: active DB session
        repo_id: the Repo this batch belongs to
        chunks_with_faiss_ids: list of dicts, each with keys matching Chunk
            columns (file_path, node_type, name, start_line, end_line,
            source_text, docstring, faiss_id)

    Returns:
        The inserted Chunk ORM objects.
    """
    rows = [
        Chunk(repo_id=repo_id, **chunk_dict)
        for chunk_dict in chunks_with_faiss_ids
    ]
    session.add_all(rows)
    session.commit()
    for row in rows:
        session.refresh(row)
    return rows


def get_chunks_by_faiss_ids(session: Session, repo_id: int, faiss_ids: list[int]) -> list[Chunk]:
    """Fetch chunk metadata rows matching the given FAISS ids, for a specific repo."""
    if not faiss_ids:
        return []
    return (
        session.query(Chunk)
        .filter(Chunk.repo_id == repo_id, Chunk.faiss_id.in_(faiss_ids))
        .all()
    )
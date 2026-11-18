"""Database connection and table definitions for JobPilot."""

from __future__ import annotations

import os

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Boolean,
    DateTime,
    func,
    text,
    ARRAY,
)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://jobpilot:jobpilot_secret@db:5432/jobpilot",
)

engine = create_async_engine(DATABASE_URL, echo=False, pool_size=10)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

VECTOR_DIM = 384  # all-MiniLM-L6-v2 output size


class Base(DeclarativeBase):
    pass


class Job(Base):
    """Single job posting with its embedding vector."""

    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(256), nullable=False)
    company = Column(String(256), nullable=False)
    location = Column(String(256), nullable=False)
    description = Column(Text, nullable=False)
    experience_years = Column(Integer, default=0)
    is_remote = Column(Boolean, default=False)
    skills = Column(ARRAY(String), default=[])
    posted_at = Column(DateTime(timezone=True), server_default=func.now())
    embedding = Column(Vector(VECTOR_DIM))


async def init_db() -> None:
    """Create pgvector extension + tables + HNSW index."""
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
        # HNSW index for fast cosine-similarity search
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_jobs_embedding_hnsw "
                "ON jobs USING hnsw (embedding vector_cosine_ops) "
                "WITH (m = 16, ef_construction = 64)"
            )
        )

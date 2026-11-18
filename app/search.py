"""Semantic search with pgvector cosine similarity + SQL filters."""

from __future__ import annotations

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import Job
from app.embeddings import embed_query
from app.schemas import SearchRequest, JobResult


async def search_jobs(session: AsyncSession, req: SearchRequest) -> list[JobResult]:
    """Embed the query, run cosine search, apply filters, return ranked results."""
    query_vec = embed_query(req.query)

    # Build cosine-distance expression: 1 - distance = similarity
    dist = Job.embedding.cosine_distance(query_vec).label("distance")

    stmt = select(Job, dist).order_by(dist)

    # Apply optional SQL filters
    if req.location:
        stmt = stmt.where(Job.location.ilike(f"%{req.location}%"))
    if req.remote_only:
        stmt = stmt.where(Job.is_remote.is_(True))
    if req.max_experience is not None:
        stmt = stmt.where(Job.experience_years <= req.max_experience)

    stmt = stmt.limit(req.top_k)

    rows = await session.execute(stmt)
    results: list[JobResult] = []
    for job, distance in rows:
        results.append(
            JobResult(
                id=job.id,
                title=job.title,
                company=job.company,
                location=job.location,
                description=job.description,
                experience_years=job.experience_years,
                is_remote=job.is_remote,
                skills=job.skills or [],
                posted_at=job.posted_at,
                similarity=round(1 - distance, 4),  # cosine similarity
            )
        )
    return results

"""Semantic search — numpy cosine similarity over SQLite-stored vectors."""

from __future__ import annotations

import json
import numpy as np

from app.db import get_conn, blob_to_vec
from app.embeddings import embed_query
from app.schemas import SearchRequest, JobResult


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalised vectors (just dot product)."""
    return float(np.dot(a, b))


def search_jobs(req: SearchRequest) -> list[JobResult]:
    """Embed the query, scan all job vectors, apply SQL filters, return top-k."""
    query_vec = embed_query(req.query)

    conn = get_conn()
    # Build WHERE clause for SQL filters
    clauses: list[str] = []
    params: list = []
    if req.location:
        clauses.append("location LIKE ?")
        params.append(f"%{req.location}%")
    if req.remote_only:
        clauses.append("is_remote = 1")
    if req.max_experience is not None:
        clauses.append("experience_years <= ?")
        params.append(req.max_experience)

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = f"SELECT * FROM jobs {where}"
    rows = conn.execute(sql, params).fetchall()
    conn.close()

    # Score every row by cosine similarity
    scored: list[tuple[dict, float]] = []
    for row in rows:
        r = dict(row)
        if r["embedding"] is None:
            continue
        job_vec = blob_to_vec(r["embedding"])
        sim = _cosine_sim(query_vec, job_vec)
        scored.append((r, sim))

    # Sort descending by similarity, take top_k
    scored.sort(key=lambda x: x[1], reverse=True)
    scored = scored[: req.top_k]

    results: list[JobResult] = []
    for r, sim in scored:
        results.append(
            JobResult(
                id=r["id"],
                title=r["title"],
                company=r["company"],
                location=r["location"],
                description=r["description"],
                experience_years=r["experience_years"],
                is_remote=bool(r["is_remote"]),
                skills=json.loads(r["skills"]) if r["skills"] else [],
                posted_at=r["posted_at"],
                similarity=round(sim, 4),
            )
        )
    return results


def search_jobs_by_ids(job_ids: list[int]) -> list[dict]:
    """Fetch specific jobs by ID list."""
    if not job_ids:
        return []
    conn = get_conn()
    placeholders = ",".join("?" for _ in job_ids)
    rows = conn.execute(
        f"SELECT * FROM jobs WHERE id IN ({placeholders})", job_ids
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

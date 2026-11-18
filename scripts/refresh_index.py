"""Incremental index refresh — insert only new postings, embed them, skip duplicates."""

from __future__ import annotations

import csv
import os
import sys
import re
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from app.db import Base, Job
from app.embeddings import embed_texts

DATABASE_URL_SYNC: str = os.getenv(
    "DATABASE_URL_SYNC",
    "postgresql+psycopg2://jobpilot:jobpilot_secret@db:5432/jobpilot",
)

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "jobs.csv"


def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def refresh() -> None:
    """Read CSV, find rows not yet in DB (by title+company+location), embed & insert."""
    if not CSV_PATH.exists():
        print("No data/jobs.csv found. Run load_jobs.py first.")
        return

    with open(CSV_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"CSV has {len(rows)} rows.")

    engine = create_engine(DATABASE_URL_SYNC)
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        # Get existing (title, company, location) tuples for dedup
        existing = set(
            session.execute(
                text("SELECT title || '||' || company || '||' || location FROM jobs")
            ).scalars().all()
        )
        print(f"DB has {len(existing)} existing jobs.")

        # Filter to new rows
        new_rows = []
        for r in rows:
            key = f"{clean_text(r['title'])}||{clean_text(r['company'])}||{clean_text(r.get('location', ''))}"
            if key not in existing:
                new_rows.append(r)

        if not new_rows:
            print("No new jobs to insert.")
            return

        print(f"Found {len(new_rows)} new jobs — embedding …")

        texts = [clean_text(f"{r['title']} — {r['description']}") for r in new_rows]
        vectors = embed_texts(texts, batch_size=128)

        for i, row in enumerate(new_rows):
            skills_list = [s.strip() for s in row.get("skills", "").split("|") if s.strip()]
            is_remote_val = row.get("is_remote", "False")
            if isinstance(is_remote_val, str):
                is_remote_val = is_remote_val.lower() in ("true", "1", "yes")

            job = Job(
                title=clean_text(row["title"]),
                company=clean_text(row["company"]),
                location=clean_text(row.get("location", "")),
                description=clean_text(row["description"]),
                experience_years=int(row.get("experience_years", 0)),
                is_remote=bool(is_remote_val),
                skills=skills_list,
                posted_at=row.get("posted_at"),
                embedding=vectors[i].tolist(),
            )
            session.add(job)

        session.commit()
        print(f"✓ Inserted {len(new_rows)} new jobs.")

    # Reindex (HNSW rebuilds automatically on INSERT, but we can REINDEX for safety)
    with engine.begin() as conn:
        conn.execute(text("REINDEX INDEX idx_jobs_embedding_hnsw"))
    print("✓ HNSW index refreshed.")


if __name__ == "__main__":
    refresh()

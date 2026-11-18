"""Incremental index refresh — insert only new postings, embed them, skip duplicates."""

from __future__ import annotations

import csv
import json
import os
import sys
import re
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

from app.db import get_conn, init_db, vec_to_blob
from app.embeddings import embed_texts

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "jobs.csv"


def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def refresh() -> None:
    """Read CSV, find rows not yet in DB (by title+company+location), embed & insert."""
    init_db()

    if not CSV_PATH.exists():
        print("No data/jobs.csv found. Run load_jobs.py first.")
        return

    with open(CSV_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"CSV has {len(rows)} rows.")

    conn = get_conn()
    # Get existing keys for dedup
    existing_rows = conn.execute("SELECT title, company, location FROM jobs").fetchall()
    existing = set(f"{r['title']}||{r['company']}||{r['location']}" for r in existing_rows)
    print(f"DB has {len(existing)} existing jobs.")

    new_rows = []
    for r in rows:
        key = f"{clean_text(r['title'])}||{clean_text(r['company'])}||{clean_text(r.get('location', ''))}"
        if key not in existing:
            new_rows.append(r)

    if not new_rows:
        print("No new jobs to insert.")
        conn.close()
        return

    print(f"Found {len(new_rows)} new jobs — embedding …")
    texts = [clean_text(f"{r['title']} — {r['description']}") for r in new_rows]
    vectors = embed_texts(texts, batch_size=128)

    for i, row in enumerate(new_rows):
        skills_list = [s.strip() for s in row.get("skills", "").split("|") if s.strip()]
        is_remote_val = row.get("is_remote", "False")
        if isinstance(is_remote_val, str):
            is_remote_val = is_remote_val.lower() in ("true", "1", "yes")

        conn.execute(
            """INSERT INTO jobs (title, company, location, description,
               experience_years, is_remote, skills, posted_at, embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                clean_text(row["title"]),
                clean_text(row["company"]),
                clean_text(row.get("location", "")),
                clean_text(row["description"]),
                int(row.get("experience_years", 0)),
                int(bool(is_remote_val)),
                json.dumps(skills_list),
                row.get("posted_at"),
                vec_to_blob(vectors[i]),
            ),
        )

    conn.commit()
    conn.close()
    print(f"✓ Inserted {len(new_rows)} new jobs.")


if __name__ == "__main__":
    refresh()

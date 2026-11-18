"""Generate eval test queries with ground-truth relevant job IDs from the actual DB.

Run AFTER load_jobs.py so the database has data.
Outputs evals/test_queries.json with 30 queries and their relevant job IDs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from app.db import get_conn, init_db


# 30 test queries covering different categories, locations, and filters
QUERIES = [
    {"query": "Python backend developer", "category": "backend", "keywords": ["Python", "Backend"]},
    {"query": "senior machine learning engineer PyTorch", "category": "ml", "keywords": ["Machine Learning", "ML", "PyTorch"]},
    {"query": "data engineer with Spark experience", "category": "data", "keywords": ["Data Engineer", "Spark"]},
    {"query": "DevOps engineer Kubernetes Docker", "category": "devops", "keywords": ["DevOps", "Kubernetes", "Docker"]},
    {"query": "React frontend developer", "category": "frontend", "keywords": ["React", "Frontend"]},
    {"query": "NLP engineer with deep learning", "category": "ml", "keywords": ["NLP", "Deep Learning"]},
    {"query": "cloud infrastructure engineer AWS", "category": "devops", "keywords": ["Cloud", "Infrastructure", "AWS"]},
    {"query": "full stack developer JavaScript", "category": "frontend", "keywords": ["Full Stack", "JavaScript"]},
    {"query": "data scientist with scikit-learn", "category": "ml", "keywords": ["Data Scientist", "scikit-learn"]},
    {"query": "site reliability engineer", "category": "devops", "keywords": ["Site Reliability", "SRE"]},
    {"query": "Go backend engineer microservices", "category": "backend", "keywords": ["Go", "Backend", "Microservices"]},
    {"query": "MLOps engineer MLflow", "category": "ml", "keywords": ["MLOps", "MLflow"]},
    {"query": "analytics engineer dbt Snowflake", "category": "data", "keywords": ["Analytics", "dbt", "Snowflake"]},
    {"query": "Vue.js frontend developer", "category": "frontend", "keywords": ["Vue.js", "Frontend"]},
    {"query": "platform engineer CI/CD", "category": "devops", "keywords": ["Platform", "CI/CD"]},
    {"query": "Java backend developer REST APIs", "category": "backend", "keywords": ["Java", "Backend", "REST"]},
    {"query": "computer vision engineer TensorFlow", "category": "ml", "keywords": ["Computer Vision", "TensorFlow"]},
    {"query": "big data engineer Kafka", "category": "data", "keywords": ["Big Data", "Kafka"]},
    {"query": "TypeScript React Next.js developer", "category": "frontend", "keywords": ["TypeScript", "React", "Next.js"]},
    {"query": "Terraform cloud engineer GCP", "category": "devops", "keywords": ["Terraform", "GCP"]},
    {"query": "API developer Python PostgreSQL", "category": "backend", "keywords": ["API", "Python", "PostgreSQL"]},
    {"query": "deep learning research engineer", "category": "ml", "keywords": ["Deep Learning", "Research"]},
    {"query": "ETL developer Airflow", "category": "data", "keywords": ["ETL", "Airflow"]},
    {"query": "Angular frontend architect", "category": "frontend", "keywords": ["Angular", "Architect"]},
    {"query": "security engineer Linux", "category": "devops", "keywords": ["Security", "Linux"]},
    {"query": "senior backend architect Redis", "category": "backend", "keywords": ["Backend", "Architect", "Redis"]},
    {"query": "AI engineer remote", "category": "ml", "keywords": ["AI Engineer"]},
    {"query": "data platform engineer BigQuery", "category": "data", "keywords": ["Data Platform", "BigQuery"]},
    {"query": "web developer HTML CSS Tailwind", "category": "frontend", "keywords": ["Web Developer", "Tailwind"]},
    {"query": "Kubernetes engineer Prometheus Grafana", "category": "devops", "keywords": ["Kubernetes", "Prometheus", "Grafana"]},
]


def find_relevant_jobs(keywords: list[str], max_results: int = 20) -> list[int]:
    """Find jobs whose title, skills, or description match the keywords."""
    conn = get_conn()
    all_jobs = conn.execute("SELECT id, title, skills, description FROM jobs").fetchall()
    conn.close()

    scored: list[tuple[int, int]] = []
    for job in all_jobs:
        score = 0
        text = f"{job['title']} {job['skills']} {job['description']}".lower()
        for kw in keywords:
            if kw.lower() in text:
                score += 1
        if score > 0:
            scored.append((job["id"], score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [jid for jid, _ in scored[:max_results]]


def generate() -> None:
    init_db()

    # Verify DB has data
    conn = get_conn()
    count = conn.execute("SELECT COUNT(*) as c FROM jobs").fetchone()["c"]
    conn.close()
    if count == 0:
        print("ERROR: Database is empty. Run scripts/load_jobs.py first.")
        return

    print(f"Database has {count} jobs. Generating test queries…")

    test_data = []
    for q in QUERIES:
        relevant_ids = find_relevant_jobs(q["keywords"])
        test_data.append({
            "query": q["query"],
            "category": q["category"],
            "relevant_job_ids": relevant_ids,
            "num_relevant": len(relevant_ids),
        })
        print(f"  '{q['query']}' → {len(relevant_ids)} relevant jobs")

    out_path = Path(__file__).parent / "test_queries.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2)
    print(f"\n✓ Saved {len(test_data)} test queries → {out_path}")


if __name__ == "__main__":
    generate()

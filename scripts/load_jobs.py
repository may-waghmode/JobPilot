"""Generate synthetic job data (if CSV is missing), embed, and bulk-insert into Postgres."""

from __future__ import annotations

import csv
import os
import random
import sys
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from app.db import Base, Job, VECTOR_DIM
from app.embeddings import embed_texts

# Use sync driver for scripts
DATABASE_URL_SYNC: str = os.getenv(
    "DATABASE_URL_SYNC",
    "postgresql+psycopg2://jobpilot:jobpilot_secret@db:5432/jobpilot",
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_PATH = DATA_DIR / "jobs.csv"

# ── Synthetic data catalogue ───────────────────────────────────────

TITLES = {
    "backend": [
        "Backend Engineer", "Senior Backend Developer", "Python Developer",
        "Go Backend Engineer", "Java Backend Developer", "Staff Backend Engineer",
        "Backend Architect", "API Developer", "Platform Engineer",
    ],
    "ml": [
        "Machine Learning Engineer", "Senior ML Engineer", "Data Scientist",
        "ML Research Engineer", "NLP Engineer", "Computer Vision Engineer",
        "AI Engineer", "Deep Learning Engineer", "MLOps Engineer",
    ],
    "data": [
        "Data Engineer", "Senior Data Engineer", "Analytics Engineer",
        "Data Platform Engineer", "ETL Developer", "Big Data Engineer",
        "Data Architect", "Database Administrator", "BI Developer",
    ],
    "devops": [
        "DevOps Engineer", "Senior DevOps Engineer", "Site Reliability Engineer",
        "Cloud Engineer", "Infrastructure Engineer", "Platform SRE",
        "Kubernetes Engineer", "Release Engineer", "Security Engineer",
    ],
    "frontend": [
        "Frontend Engineer", "Senior Frontend Developer", "React Developer",
        "Full Stack Developer", "UI Engineer", "Frontend Architect",
        "Angular Developer", "Vue.js Developer", "Web Developer",
    ],
}

SKILLS_MAP = {
    "backend": ["Python", "Go", "Java", "REST APIs", "PostgreSQL", "Redis", "Docker", "Microservices", "gRPC", "Kafka"],
    "ml": ["Python", "PyTorch", "TensorFlow", "scikit-learn", "NLP", "Computer Vision", "MLflow", "Pandas", "SQL", "Kubernetes"],
    "data": ["Python", "SQL", "Spark", "Airflow", "dbt", "Snowflake", "Kafka", "BigQuery", "AWS Glue", "Redshift"],
    "devops": ["Kubernetes", "Docker", "Terraform", "AWS", "GCP", "CI/CD", "Ansible", "Prometheus", "Grafana", "Linux"],
    "frontend": ["JavaScript", "TypeScript", "React", "Next.js", "HTML/CSS", "Tailwind CSS", "GraphQL", "Vue.js", "Webpack", "Jest"],
}

COMPANIES = [
    "TechNova", "DataMind AI", "CloudSprint", "NexGen Labs", "CodeCraft",
    "InfiniStack", "BrightPath", "ScaleForce", "QuantumLeap", "PivotCore",
    "ZenithOps", "Synthwave Systems", "MeshGrid", "PulseTech", "Orbitix",
    "FusionWare", "DeepRoot AI", "ClearStack", "ByteShift", "EdgePoint",
    "FlowState", "CatalystX", "LogicLoom", "NeuralNode", "GridIron Tech",
    "VelocityAI", "HyperLoop Labs", "SpectrumDB", "CoreAxis", "BlueShift",
]

LOCATIONS = ["Bengaluru", "Gurugram", "Pune", "Hyderabad", "Remote"]

DESCRIPTION_TEMPLATES = [
    "We are looking for a {title} to join our {team} team. You will {action} and work closely with {collab}. Requirements: {exp}+ years of experience with {skills_str}.",
    "Join {company} as a {title}! In this role, you'll {action}. We need someone with {exp}+ years of hands-on experience in {skills_str}. {perk}",
    "{company} is hiring a {title} ({location}). You will {action}, collaborating with {collab}. Must have: {skills_str}. Experience: {exp}+ years. {perk}",
    "Exciting opportunity for a {title} at {company}. Responsibilities include {action_long}. Strong proficiency in {skills_str} required ({exp}+ years). {perk}",
]

ACTIONS = [
    "design and build scalable services", "develop high-performance APIs",
    "architect distributed systems", "build data pipelines at scale",
    "train and deploy ML models in production", "optimize system reliability",
    "create responsive and accessible UIs", "implement CI/CD pipelines",
    "lead technical design discussions", "mentor junior engineers",
]

ACTIONS_LONG = [
    "designing, implementing, and maintaining production services that handle millions of requests daily",
    "building end-to-end ML pipelines from data ingestion through model serving",
    "creating robust infrastructure-as-code and automating deployment workflows",
    "developing and optimising frontend applications for speed and accessibility",
    "leading the data platform roadmap and driving engineering best practices",
]

COLLABS = ["cross-functional teams", "product and design", "data science and engineering", "platform and infra teams"]
TEAMS = ["engineering", "platform", "product", "growth", "infrastructure", "AI/ML"]
PERKS = [
    "Competitive salary + ESOPs.", "Flexible work hours and remote-friendly culture.",
    "Learning budget and conference sponsorship.", "Health insurance for you and family.",
    "Work on cutting-edge technology.", "",
]


def generate_synthetic_csv(n: int = 2000) -> None:
    """Write ~n synthetic job rows to data/jobs.csv."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    categories = list(TITLES.keys())

    for i in range(n):
        cat = random.choice(categories)
        title = random.choice(TITLES[cat])
        company = random.choice(COMPANIES)
        location = random.choice(LOCATIONS)
        is_remote = location == "Remote" or random.random() < 0.15
        exp = random.randint(0, 12)
        skills = random.sample(SKILLS_MAP[cat], k=random.randint(3, 6))
        days_ago = random.randint(0, 90)
        posted = datetime.now() - timedelta(days=days_ago)

        desc = random.choice(DESCRIPTION_TEMPLATES).format(
            title=title,
            company=company,
            location=location,
            team=random.choice(TEAMS),
            action=random.choice(ACTIONS),
            action_long=random.choice(ACTIONS_LONG),
            collab=random.choice(COLLABS),
            skills_str=", ".join(skills),
            exp=exp,
            perk=random.choice(PERKS),
        )

        rows.append(
            {
                "title": title,
                "company": company,
                "location": location,
                "description": desc,
                "experience_years": exp,
                "is_remote": is_remote,
                "skills": "|".join(skills),
                "posted_at": posted.isoformat(),
            }
        )

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"✓ Generated {len(rows)} synthetic jobs → {CSV_PATH}")


def clean_text(s: str) -> str:
    """Collapse whitespace and strip."""
    return re.sub(r"\s+", " ", s).strip()


def load() -> None:
    """Main loader: generate CSV if needed, embed, insert."""
    if not CSV_PATH.exists():
        print("jobs.csv not found — generating synthetic data …")
        generate_synthetic_csv()

    # Read CSV
    with open(CSV_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"Read {len(rows)} rows from {CSV_PATH}")

    # Prepare text for embedding: title + description
    texts = [clean_text(f"{r['title']} — {r['description']}") for r in rows]

    # Batch-embed
    print("Embedding …")
    vectors = embed_texts(texts, batch_size=128)

    # Insert into Postgres
    engine = create_engine(DATABASE_URL_SYNC)

    # Ensure extension + tables exist
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    Base.metadata.create_all(engine)

    # Create HNSW index
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_jobs_embedding_hnsw "
                "ON jobs USING hnsw (embedding vector_cosine_ops) "
                "WITH (m = 16, ef_construction = 64)"
            )
        )

    with Session(engine) as session:
        # Truncate for idempotent loads
        session.execute(text("TRUNCATE TABLE jobs RESTART IDENTITY"))
        session.commit()

        for i, row in enumerate(rows):
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

            if (i + 1) % 500 == 0:
                session.commit()
                print(f"  Inserted {i + 1}/{len(rows)} …")

        session.commit()
        print(f"✓ Loaded {len(rows)} jobs into Postgres.")


if __name__ == "__main__":
    load()

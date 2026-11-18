"""Resume PDF parsing → structured profile via LLM."""

from __future__ import annotations

import json
import os
from datetime import datetime

import pymupdf as fitz  # PyMuPDF

from app.db import get_conn
from app.llm import llm_structured_extract
from app.schemas import ParsedProfile


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract all text from a PDF file."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n\n".join(pages).strip()


def parse_resume(pdf_bytes: bytes, user_id: str = "default") -> ParsedProfile:
    """Extract text from PDF, call LLM for structured profile, store in DB."""
    raw_text = extract_text_from_pdf(pdf_bytes)
    if not raw_text:
        raise ValueError("Could not extract any text from the PDF.")

    # LLM structured extraction
    profile = llm_structured_extract(raw_text)

    # Persist to DB
    conn = get_conn()
    # Upsert: delete old profile for this user, insert new one
    conn.execute("DELETE FROM user_profiles WHERE user_id = ?", (user_id,))
    conn.execute(
        """INSERT INTO user_profiles
           (user_id, skills, years_experience, roles, preferred_locations, raw_text, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            user_id,
            json.dumps(profile.skills),
            profile.years_experience,
            json.dumps(profile.roles),
            json.dumps(profile.preferred_locations),
            raw_text,
            datetime.now().isoformat(),
        ),
    )
    conn.commit()
    conn.close()
    return profile

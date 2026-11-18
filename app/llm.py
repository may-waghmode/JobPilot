"""LLM abstraction — supports Google Gemini and OpenAI, configurable via .env."""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "google").lower()
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")


# ── Structured extraction (resume → ParsedProfile) ─────────────

EXTRACT_PROMPT = """You are a resume parser. Given the raw text of a resume,
extract a structured profile with these exact fields:
- skills: list of technical and soft skills mentioned
- years_experience: total years of professional experience (integer)
- roles: list of job titles / roles the person has held
- preferred_locations: list of locations mentioned or inferred (cities)

Return ONLY valid JSON matching this schema, no markdown fences:
{{"skills": [...], "years_experience": N, "roles": [...], "preferred_locations": [...]}}

Resume text:
{text}"""


def llm_structured_extract(resume_text: str) -> "ParsedProfile":
    """Call LLM to extract structured profile from resume text."""
    from app.schemas import ParsedProfile

    prompt = EXTRACT_PROMPT.format(text=resume_text[:8000])  # cap length
    raw = _call_llm(prompt)
    # Parse JSON from response
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    data = json.loads(raw)
    return ParsedProfile(**data)


# ── Chat completion (agent loop) ───────────────────────────────

def llm_chat(messages: list[dict[str, str]], tools_desc: str = "") -> str:
    """Send a multi-turn conversation to the LLM and get a text reply."""
    system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
    user_msgs = [m for m in messages if m["role"] != "system"]

    # Build a single prompt from messages (simple but effective)
    parts = []
    if system:
        parts.append(f"System: {system}")
    if tools_desc:
        parts.append(f"Available tools:\n{tools_desc}")
    for m in user_msgs:
        parts.append(f"{m['role'].capitalize()}: {m['content']}")
    parts.append("Assistant:")
    prompt = "\n\n".join(parts)
    return _call_llm(prompt)


# ── LLM judge for evals ────────────────────────────────────────

def llm_judge(query: str, response: str) -> dict[str, Any]:
    """Use LLM to judge chat response quality. Returns {score, reasoning}."""
    prompt = f"""You are evaluating an AI job-search assistant's response.
Rate the response on a scale of 1-5 for:
- relevance: Does it address the user's query?
- helpfulness: Does it provide actionable job recommendations?
- coherence: Is the response well-structured?

User query: {query}
Assistant response: {response}

Return ONLY valid JSON: {{"relevance": N, "helpfulness": N, "coherence": N, "reasoning": "..."}}"""
    raw = _call_llm(prompt)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"relevance": 3, "helpfulness": 3, "coherence": 3, "reasoning": "Parse error"}


# ── Provider dispatch ──────────────────────────────────────────

def _call_llm(prompt: str) -> str:
    """Route to the configured LLM provider."""
    if LLM_PROVIDER == "google":
        return _call_google(prompt)
    elif LLM_PROVIDER == "openai":
        return _call_openai(prompt)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")


def _call_google(prompt: str) -> str:
    """Call Google Gemini API."""
    from google import genai

    client = genai.Client(api_key=GOOGLE_API_KEY)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    return response.text or ""


def _call_openai(prompt: str) -> str:
    """Call OpenAI API."""
    import openai

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content or ""

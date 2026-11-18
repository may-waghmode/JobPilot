"""Pydantic schemas for requests, responses, and structured LLM output."""

from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, Field


# ── Search ──────────────────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Free-text job search query")
    location: str | None = Field(None, description="Filter by location substring")
    remote_only: bool = Field(False, description="Show only remote jobs")
    max_experience: int | None = Field(None, ge=0, description="Max years of experience")
    top_k: int = Field(10, ge=1, le=100, description="Number of results")


class JobResult(BaseModel):
    id: int
    title: str
    company: str
    location: str
    description: str
    experience_years: int
    is_remote: bool
    skills: list[str]
    posted_at: str | None
    similarity: float = Field(..., description="Cosine similarity score (0–1)")


class SearchResponse(BaseModel):
    query: str
    count: int
    results: list[JobResult]


# ── Resume / Profile ───────────────────────────────────────────
class ParsedProfile(BaseModel):
    """Structured output the LLM must return when parsing a resume."""
    skills: list[str] = Field(default_factory=list, description="Technical and soft skills")
    years_experience: int = Field(0, description="Total years of professional experience")
    roles: list[str] = Field(default_factory=list, description="Job titles / roles held")
    preferred_locations: list[str] = Field(default_factory=list, description="Preferred work locations")


class ProfileResponse(BaseModel):
    user_id: str
    profile: ParsedProfile
    message: str


# ── Chat / Agent ────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    user_id: str = Field("default")


class ToolCallInfo(BaseModel):
    tool: str
    args: dict
    result_preview: str = ""


class ChatResponse(BaseModel):
    reply: str
    tool_calls: list[ToolCallInfo] = []
    recommended_jobs: list[JobResult] = []

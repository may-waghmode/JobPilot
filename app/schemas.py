"""Pydantic schemas for request / response validation."""

from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, Field


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
    posted_at: datetime | None
    similarity: float = Field(..., description="Cosine similarity score (0-1)")


class SearchResponse(BaseModel):
    query: str
    count: int
    results: list[JobResult]

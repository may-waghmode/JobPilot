"""FastAPI application — API + static UI for JobPilot."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.db import init_db
from app.schemas import (
    SearchRequest, SearchResponse,
    ChatRequest, ChatResponse,
    ProfileResponse, ParsedProfile,
)
from app.search import search_jobs
from app.agent import run_agent
from app.profile import parse_resume

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise database on startup."""
    init_db()
    yield


app = FastAPI(title="JobPilot", version="0.2.0", lifespan=lifespan)

# Mount static assets
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Health ──────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Semantic Search ─────────────────────────────────────────────
@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    results = search_jobs(req)
    return SearchResponse(query=req.query, count=len(results), results=results)


# ── Resume Upload ───────────────────────────────────────────────
@app.post("/profile/upload", response_model=ProfileResponse)
async def upload_profile(
    file: UploadFile = File(...),
    user_id: str = Form("default"),
):
    pdf_bytes = await file.read()
    profile = parse_resume(pdf_bytes, user_id)
    return ProfileResponse(
        user_id=user_id,
        profile=profile,
        message="Resume parsed and profile saved successfully.",
    )


# ── Chat (Agent) ───────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    return run_agent(req.message, req.user_id)


# ── UI ──────────────────────────────────────────────────────────
@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))

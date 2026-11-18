"""FastAPI application — API + static UI for JobPilot."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.db import async_session, init_db
from app.schemas import SearchRequest, SearchResponse
from app.search import search_jobs

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise database on startup."""
    await init_db()
    yield


app = FastAPI(title="JobPilot", version="0.1.0", lifespan=lifespan)

# Mount static assets (CSS/JS if any)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Health ──────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Semantic Search ─────────────────────────────────────────────────
@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    async with async_session() as session:
        results = await search_jobs(session, req)
    return SearchResponse(query=req.query, count=len(results), results=results)


# ── UI ──────────────────────────────────────────────────────────────
@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))

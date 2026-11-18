"""Database layer — SQLite with JSON columns for arrays, numpy for vectors."""

from __future__ import annotations

import json
import sqlite3
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DB_PATH: str = os.getenv("DATABASE_PATH", "data/jobpilot.db")


def _ensure_dir() -> None:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)


def get_conn() -> sqlite3.Connection:
    """Return a new connection with row-factory enabled."""
    _ensure_dir()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    """Create all tables if they don't exist."""
    conn = get_conn()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            title       TEXT    NOT NULL,
            company     TEXT    NOT NULL,
            location    TEXT    NOT NULL DEFAULT '',
            description TEXT    NOT NULL,
            experience_years INTEGER DEFAULT 0,
            is_remote   INTEGER DEFAULT 0,
            skills      TEXT    DEFAULT '[]',
            posted_at   TEXT,
            embedding   BLOB
        );

        CREATE TABLE IF NOT EXISTS user_profiles (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id             TEXT    NOT NULL DEFAULT 'default',
            skills              TEXT    DEFAULT '[]',
            years_experience    INTEGER DEFAULT 0,
            roles               TEXT    DEFAULT '[]',
            preferred_locations TEXT    DEFAULT '[]',
            raw_text            TEXT    DEFAULT '',
            created_at          TEXT
        );

        CREATE TABLE IF NOT EXISTS user_preferences (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     TEXT    NOT NULL DEFAULT 'default',
            key         TEXT    NOT NULL,
            value       TEXT    NOT NULL,
            updated_at  TEXT
        );

        CREATE TABLE IF NOT EXISTS conversation_memory (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     TEXT    NOT NULL DEFAULT 'default',
            summary     TEXT    DEFAULT '',
            messages    TEXT    DEFAULT '[]',
            updated_at  TEXT
        );
        """
    )
    conn.commit()
    conn.close()


# ── Helper: serialize / deserialize numpy arrays as bytes ───────
import numpy as np


def vec_to_blob(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32).tobytes()


def blob_to_vec(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)

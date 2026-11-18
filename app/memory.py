"""User preferences and conversation memory — stored in SQLite."""

from __future__ import annotations

import json
from datetime import datetime

from app.db import get_conn


# ── Preferences ─────────────────────────────────────────────────

def get_preferences(user_id: str = "default") -> dict[str, str]:
    """Load all preferences for a user as a dict."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT key, value FROM user_preferences WHERE user_id = ?", (user_id,)
    ).fetchall()
    conn.close()
    return {r["key"]: r["value"] for r in rows}


def save_preference(user_id: str, key: str, value: str) -> None:
    """Upsert a single preference."""
    conn = get_conn()
    existing = conn.execute(
        "SELECT id FROM user_preferences WHERE user_id = ? AND key = ?",
        (user_id, key),
    ).fetchone()
    now = datetime.now().isoformat()
    if existing:
        conn.execute(
            "UPDATE user_preferences SET value = ?, updated_at = ? WHERE id = ?",
            (value, now, existing["id"]),
        )
    else:
        conn.execute(
            "INSERT INTO user_preferences (user_id, key, value, updated_at) VALUES (?, ?, ?, ?)",
            (user_id, key, value, now),
        )
    conn.commit()
    conn.close()


# ── Conversation Memory ────────────────────────────────────────

def get_memory(user_id: str = "default") -> dict:
    """Load rolling conversation summary + recent messages."""
    conn = get_conn()
    row = conn.execute(
        "SELECT summary, messages FROM conversation_memory WHERE user_id = ? ORDER BY id DESC LIMIT 1",
        (user_id,),
    ).fetchone()
    conn.close()
    if row:
        return {
            "summary": row["summary"],
            "messages": json.loads(row["messages"]) if row["messages"] else [],
        }
    return {"summary": "", "messages": []}


def update_memory(user_id: str, summary: str, messages: list[dict]) -> None:
    """Upsert the conversation memory — keep last 20 messages."""
    messages = messages[-20:]  # rolling window
    conn = get_conn()
    existing = conn.execute(
        "SELECT id FROM conversation_memory WHERE user_id = ?", (user_id,)
    ).fetchone()
    now = datetime.now().isoformat()
    msgs_json = json.dumps(messages)
    if existing:
        conn.execute(
            "UPDATE conversation_memory SET summary = ?, messages = ?, updated_at = ? WHERE id = ?",
            (summary, msgs_json, now, existing["id"]),
        )
    else:
        conn.execute(
            "INSERT INTO conversation_memory (user_id, summary, messages, updated_at) VALUES (?, ?, ?, ?)",
            (user_id, summary, msgs_json, now),
        )
    conn.commit()
    conn.close()


def get_profile_dict(user_id: str = "default") -> dict | None:
    """Load the user's parsed profile as a dict."""
    conn = get_conn()
    row = conn.execute(
        "SELECT skills, years_experience, roles, preferred_locations FROM user_profiles WHERE user_id = ? ORDER BY id DESC LIMIT 1",
        (user_id,),
    ).fetchone()
    conn.close()
    if not row:
        return None
    return {
        "skills": json.loads(row["skills"]) if row["skills"] else [],
        "years_experience": row["years_experience"],
        "roles": json.loads(row["roles"]) if row["roles"] else [],
        "preferred_locations": json.loads(row["preferred_locations"]) if row["preferred_locations"] else [],
    }

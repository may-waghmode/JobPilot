"""LLM tool-calling agent with search, profile, preference and ranking tools."""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np

from app.db import get_conn, blob_to_vec
from app.embeddings import embed_query
from app.llm import llm_chat
from app.memory import (
    get_preferences,
    save_preference,
    get_memory,
    update_memory,
    get_profile_dict,
)
from app.schemas import (
    SearchRequest,
    JobResult,
    ToolCallInfo,
    ChatResponse,
)
from app.search import search_jobs


# ── Tool definitions (shown to the LLM) ────────────────────────

TOOLS_DESC = """
You have these tools. To call one, reply with EXACTLY this format on its own line:
TOOL_CALL: tool_name({"param": "value", ...})

Tools:
1. search_jobs({"query": "...", "location": "...", "remote_only": false, "max_experience": null, "top_k": 10})
   — Semantic job search with optional filters. Returns matched jobs.

2. get_user_profile()
   — Returns the user's parsed resume profile (skills, experience, roles, locations).

3. save_preference({"key": "...", "value": "..."})
   — Save a user preference (e.g. key="preferred_role", value="backend engineer").

4. rank_jobs({"job_ids": [1, 2, 3]})
   — Hybrid-rank specific jobs using embedding similarity + skill overlap + preferences. Returns re-ranked list.

After getting tool results, synthesize a helpful response for the user.
If the user asks about jobs, ALWAYS use search_jobs first.
If the user mentions their background, use get_user_profile to personalize.
You may call multiple tools in sequence by responding with multiple TOOL_CALL lines.
When you have enough information, provide your final answer WITHOUT any TOOL_CALL lines.
"""

SYSTEM_PROMPT = """You are JobPilot, an AI job-search assistant.
You help users find relevant tech jobs by understanding their skills, experience, and preferences.
You have access to a database of job postings and can search semantically.
Be concise, helpful, and always try to recommend specific jobs when relevant.
If the user has uploaded a resume, use their profile to personalize recommendations.
{memory_context}
{prefs_context}
{profile_context}
"""


# ── Tool execution ──────────────────────────────────────────────

def _exec_search_jobs(args: dict) -> tuple[str, list[JobResult]]:
    """Execute search_jobs tool."""
    req = SearchRequest(
        query=args.get("query", ""),
        location=args.get("location"),
        remote_only=args.get("remote_only", False),
        max_experience=args.get("max_experience"),
        top_k=args.get("top_k", 10),
    )
    results = search_jobs(req)
    # Format for LLM context
    summaries = []
    for j in results[:10]:
        summaries.append(
            f"[ID:{j.id}] {j.title} at {j.company} ({j.location}) "
            f"| {j.experience_years}yr | Skills: {', '.join(j.skills[:5])} "
            f"| Sim: {j.similarity}"
        )
    return "\n".join(summaries) if summaries else "No jobs found.", results


def _exec_get_profile(user_id: str) -> str:
    profile = get_profile_dict(user_id)
    if not profile:
        return "No resume profile uploaded yet."
    return json.dumps(profile, indent=2)


def _exec_save_preference(user_id: str, args: dict) -> str:
    key = args.get("key", "")
    value = args.get("value", "")
    if not key or not value:
        return "Error: key and value are required."
    save_preference(user_id, key, value)
    return f"Preference saved: {key} = {value}"


def _exec_rank_jobs(user_id: str, args: dict) -> tuple[str, list[JobResult]]:
    """Hybrid ranking: embedding sim + skill overlap + preference boost."""
    job_ids = args.get("job_ids", [])
    if not job_ids:
        return "No job IDs provided.", []

    profile = get_profile_dict(user_id) or {}
    prefs = get_preferences(user_id)
    profile_skills = set(s.lower() for s in profile.get("skills", []))
    pref_role = prefs.get("preferred_role", "").lower()
    pref_location = prefs.get("preferred_location", "").lower()

    # Build a query from profile + preferences for embedding similarity
    query_parts = list(profile.get("roles", [])) + list(profile_skills)[:5]
    if pref_role:
        query_parts.insert(0, pref_role)
    query_text = " ".join(query_parts) if query_parts else "software engineer"
    query_vec = embed_query(query_text)

    conn = get_conn()
    placeholders = ",".join("?" for _ in job_ids)
    rows = conn.execute(
        f"SELECT * FROM jobs WHERE id IN ({placeholders})", job_ids
    ).fetchall()
    conn.close()

    scored: list[tuple[dict, float]] = []
    for row in rows:
        r = dict(row)
        job_skills = set(s.lower() for s in json.loads(r["skills"])) if r["skills"] else set()

        # 1. Embedding similarity (0–1)
        emb_sim = 0.0
        if r["embedding"]:
            job_vec = blob_to_vec(r["embedding"])
            emb_sim = float(np.dot(query_vec, job_vec))

        # 2. Skill overlap (Jaccard-ish, 0–1)
        skill_overlap = 0.0
        if profile_skills and job_skills:
            skill_overlap = len(profile_skills & job_skills) / len(profile_skills | job_skills)

        # 3. Preference boost
        pref_boost = 0.0
        if pref_role and pref_role in r["title"].lower():
            pref_boost += 0.1
        if pref_location and pref_location in r["location"].lower():
            pref_boost += 0.05

        # Weighted hybrid score
        hybrid = 0.5 * emb_sim + 0.35 * skill_overlap + 0.15 * min(pref_boost / 0.15, 1.0)
        scored.append((r, round(hybrid, 4)))

    scored.sort(key=lambda x: x[1], reverse=True)

    results: list[JobResult] = []
    summaries: list[str] = []
    for r, score in scored:
        jr = JobResult(
            id=r["id"],
            title=r["title"],
            company=r["company"],
            location=r["location"],
            description=r["description"],
            experience_years=r["experience_years"],
            is_remote=bool(r["is_remote"]),
            skills=json.loads(r["skills"]) if r["skills"] else [],
            posted_at=r["posted_at"],
            similarity=score,
        )
        results.append(jr)
        summaries.append(
            f"[ID:{jr.id}] {jr.title} at {jr.company} | Hybrid: {score}"
        )

    return "\n".join(summaries) if summaries else "No jobs found.", results


# ── Agent loop ──────────────────────────────────────────────────

TOOL_CALL_RE = re.compile(r'TOOL_CALL:\s*(\w+)\((\{.*?\})\)', re.DOTALL)

MAX_TOOL_ROUNDS = 4


def run_agent(user_message: str, user_id: str = "default") -> ChatResponse:
    """Execute the agent loop: build context → LLM → tool calls → response."""
    # Load memory, preferences, profile
    memory = get_memory(user_id)
    prefs = get_preferences(user_id)
    profile = get_profile_dict(user_id)

    memory_ctx = ""
    if memory["summary"]:
        memory_ctx = f"Previous conversation summary: {memory['summary']}"
    prefs_ctx = ""
    if prefs:
        prefs_ctx = f"User preferences: {json.dumps(prefs)}"
    profile_ctx = ""
    if profile:
        profile_ctx = f"User profile: {json.dumps(profile)}"

    system = SYSTEM_PROMPT.format(
        memory_context=memory_ctx,
        prefs_context=prefs_ctx,
        profile_context=profile_ctx,
    )

    # Build conversation messages
    messages = [{"role": "system", "content": system}]
    # Add recent memory messages
    for m in memory.get("messages", [])[-6:]:
        messages.append(m)
    messages.append({"role": "user", "content": user_message})

    all_tool_calls: list[ToolCallInfo] = []
    all_recommended: list[JobResult] = []

    for _round in range(MAX_TOOL_ROUNDS):
        reply = llm_chat(messages, tools_desc=TOOLS_DESC)

        # Check for tool calls
        tool_matches = TOOL_CALL_RE.findall(reply)
        if not tool_matches:
            # No tool calls — this is the final answer
            break

        # Execute each tool call
        tool_results = []
        for tool_name, args_str in tool_matches:
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                args = {}

            result_text = ""
            if tool_name == "search_jobs":
                result_text, jobs = _exec_search_jobs(args)
                all_recommended.extend(jobs)
            elif tool_name == "get_user_profile":
                result_text = _exec_get_profile(user_id)
            elif tool_name == "save_preference":
                result_text = _exec_save_preference(user_id, args)
            elif tool_name == "rank_jobs":
                result_text, jobs = _exec_rank_jobs(user_id, args)
                all_recommended = jobs  # replace with ranked
            else:
                result_text = f"Unknown tool: {tool_name}"

            all_tool_calls.append(ToolCallInfo(
                tool=tool_name,
                args=args,
                result_preview=result_text[:200],
            ))
            tool_results.append(f"[Tool: {tool_name}] Result:\n{result_text}")

        # Feed tool results back to LLM
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": "\n\n".join(tool_results) + "\n\nNow provide your final answer to the user based on these results."})
    else:
        reply = "I found some results for you. Please see the recommended jobs below."

    # Clean tool call markers from final reply
    final_reply = TOOL_CALL_RE.sub("", reply).strip()
    if not final_reply:
        final_reply = "Here are the jobs I found for you based on your query."

    # Deduplicate recommended jobs by ID
    seen_ids: set[int] = set()
    unique_jobs: list[JobResult] = []
    for j in all_recommended:
        if j.id not in seen_ids:
            seen_ids.add(j.id)
            unique_jobs.append(j)

    # Update memory
    new_messages = memory.get("messages", [])
    new_messages.append({"role": "user", "content": user_message})
    new_messages.append({"role": "assistant", "content": final_reply[:500]})
    # Simple summary: last user message
    summary = f"Last topic: {user_message[:200]}"
    if memory.get("summary"):
        summary = f"{memory['summary']} | {summary}"
    summary = summary[-500:]  # cap length
    update_memory(user_id, summary, new_messages)

    return ChatResponse(
        reply=final_reply,
        tool_calls=all_tool_calls,
        recommended_jobs=unique_jobs[:10],
    )

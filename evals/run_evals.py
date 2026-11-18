"""Evaluation script — computes Precision@5 and MRR for vector-only and hybrid search.

Also runs LLM-as-judge for chat quality (if LLM API key is configured).
Outputs results to evals/results/ and stdout.
"""

from __future__ import annotations

import json
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import numpy as np

from app.db import get_conn, init_db, blob_to_vec
from app.embeddings import embed_query
from app.schemas import SearchRequest
from app.search import search_jobs

EVALS_DIR = Path(__file__).parent
RESULTS_DIR = EVALS_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_test_queries() -> list[dict]:
    path = EVALS_DIR / "test_queries.json"
    if not path.exists():
        print("ERROR: test_queries.json not found. Run generate_test_queries.py first.")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Metrics ─────────────────────────────────────────────────────

def precision_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int = 5) -> float:
    """Fraction of top-k retrieved that are relevant."""
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    return len(set(top_k) & relevant_ids) / k


def reciprocal_rank(retrieved_ids: list[int], relevant_ids: set[int]) -> float:
    """1/rank of the first relevant result."""
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


# ── Vector-only search evaluation ──────────────────────────────

def eval_vector_search(test_queries: list[dict]) -> dict:
    """Evaluate pure cosine-similarity vector search."""
    p5_scores = []
    mrr_scores = []

    for tq in test_queries:
        req = SearchRequest(query=tq["query"], top_k=20)
        results = search_jobs(req)
        retrieved = [r.id for r in results]
        relevant = set(tq["relevant_job_ids"])

        p5 = precision_at_k(retrieved, relevant, k=5)
        rr = reciprocal_rank(retrieved, relevant)
        p5_scores.append(p5)
        mrr_scores.append(rr)

    return {
        "method": "Vector-only (cosine)",
        "mean_precision_at_5": round(np.mean(p5_scores), 4),
        "mean_mrr": round(np.mean(mrr_scores), 4),
        "per_query": [
            {
                "query": tq["query"],
                "p@5": round(p5_scores[i], 4),
                "rr": round(mrr_scores[i], 4),
            }
            for i, tq in enumerate(test_queries)
        ],
    }


# ── Hybrid search evaluation ──────────────────────────────────

def hybrid_search(query: str, top_k: int = 20) -> list[tuple[int, float]]:
    """Hybrid: 0.6 * embedding sim + 0.4 * keyword overlap score."""
    query_vec = embed_query(query)
    query_terms = set(query.lower().split())

    conn = get_conn()
    rows = conn.execute("SELECT id, title, skills, description, embedding FROM jobs").fetchall()
    conn.close()

    scored = []
    for row in rows:
        if row["embedding"] is None:
            continue
        job_vec = blob_to_vec(row["embedding"])
        emb_sim = float(np.dot(query_vec, job_vec))

        # Keyword overlap score
        text = f"{row['title']} {row['skills']} {row['description']}".lower()
        text_terms = set(text.split())
        if query_terms:
            keyword_score = len(query_terms & text_terms) / len(query_terms)
        else:
            keyword_score = 0.0

        hybrid = 0.6 * emb_sim + 0.4 * keyword_score
        scored.append((row["id"], round(hybrid, 4)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def eval_hybrid_search(test_queries: list[dict]) -> dict:
    """Evaluate hybrid (embedding + keyword) search."""
    p5_scores = []
    mrr_scores = []

    for tq in test_queries:
        results = hybrid_search(tq["query"], top_k=20)
        retrieved = [r[0] for r in results]
        relevant = set(tq["relevant_job_ids"])

        p5 = precision_at_k(retrieved, relevant, k=5)
        rr = reciprocal_rank(retrieved, relevant)
        p5_scores.append(p5)
        mrr_scores.append(rr)

    return {
        "method": "Hybrid (embedding + keyword)",
        "mean_precision_at_5": round(np.mean(p5_scores), 4),
        "mean_mrr": round(np.mean(mrr_scores), 4),
        "per_query": [
            {
                "query": tq["query"],
                "p@5": round(p5_scores[i], 4),
                "rr": round(mrr_scores[i], 4),
            }
            for i, tq in enumerate(test_queries)
        ],
    }


# ── LLM-as-Judge (chat quality) ───────────────────────────────

def eval_llm_judge(test_queries: list[dict], n: int = 5) -> dict | None:
    """Use LLM to judge chat response quality (runs only if API key is set)."""
    api_key = os.getenv("GOOGLE_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("your-"):
        print("⚠ Skipping LLM-as-judge: no valid API key configured.")
        return None

    from app.llm import llm_judge
    from app.agent import run_agent

    scores = []
    for tq in test_queries[:n]:
        try:
            resp = run_agent(tq["query"], user_id="eval_user")
            judgment = llm_judge(tq["query"], resp.reply)
            judgment["query"] = tq["query"]
            scores.append(judgment)
            print(f"  Judge: '{tq['query'][:40]}…' → rel={judgment.get('relevance')}, help={judgment.get('helpfulness')}, coh={judgment.get('coherence')}")
        except Exception as e:
            print(f"  Judge error on '{tq['query'][:40]}': {e}")

    if not scores:
        return None

    return {
        "method": "LLM-as-Judge (chat quality)",
        "num_evaluated": len(scores),
        "mean_relevance": round(np.mean([s.get("relevance", 3) for s in scores]), 2),
        "mean_helpfulness": round(np.mean([s.get("helpfulness", 3) for s in scores]), 2),
        "mean_coherence": round(np.mean([s.get("coherence", 3) for s in scores]), 2),
        "details": scores,
    }


# ── Main ───────────────────────────────────────────────────────

def run_evals() -> None:
    init_db()
    test_queries = load_test_queries()
    print(f"Loaded {len(test_queries)} test queries.\n")

    # 1. Vector-only search
    print("═══ Vector-Only Search ═══")
    vec_results = eval_vector_search(test_queries)
    print(f"  Precision@5: {vec_results['mean_precision_at_5']}")
    print(f"  MRR:         {vec_results['mean_mrr']}")

    # 2. Hybrid search
    print("\n═══ Hybrid Search ═══")
    hyb_results = eval_hybrid_search(test_queries)
    print(f"  Precision@5: {hyb_results['mean_precision_at_5']}")
    print(f"  MRR:         {hyb_results['mean_mrr']}")

    # 3. LLM-as-Judge
    print("\n═══ LLM-as-Judge ═══")
    judge_results = eval_llm_judge(test_queries)

    # Save all results
    all_results = {
        "vector_search": vec_results,
        "hybrid_search": hyb_results,
    }
    if judge_results:
        all_results["llm_judge"] = judge_results

    out_path = RESULTS_DIR / "eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Method':<35} {'P@5':>8} {'MRR':>8}")
    print("-" * 60)
    print(f"{'Vector-only (cosine)':<35} {vec_results['mean_precision_at_5']:>8.4f} {vec_results['mean_mrr']:>8.4f}")
    print(f"{'Hybrid (embedding + keyword)':<35} {hyb_results['mean_precision_at_5']:>8.4f} {hyb_results['mean_mrr']:>8.4f}")
    if judge_results:
        print(f"\nLLM-as-Judge (n={judge_results['num_evaluated']}): "
              f"Relevance={judge_results['mean_relevance']}/5, "
              f"Helpfulness={judge_results['mean_helpfulness']}/5, "
              f"Coherence={judge_results['mean_coherence']}/5")
    print("=" * 60)
    print(f"\n✓ Full results saved → {out_path}")


if __name__ == "__main__":
    run_evals()

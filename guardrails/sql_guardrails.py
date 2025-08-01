
from __future__ import annotations
import os, re
from typing import List, Tuple, Any, Sequence
import numpy as np
from sentence_transformers import CrossEncoder


_CE_NAME = os.getenv("CE_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
ce_model = CrossEncoder(_CE_NAME)           # ~110 MB, loads once

_NUM_RE = re.compile(r"-?\d+\.?\d*")

def _relevance(q: str, a: str) -> float:
    """Cross‑encoder score → [0,1]."""
    raw = ce_model.predict([(q, a)])[0]      # MS‑Marco 0‑4 range
    return max(min(raw / 4.0, 1.0), 0.0)

def _length_score(ans: str, min_len=20, max_len=600) -> float:
    n = len(ans.split())
    if n < min_len:
        return n / min_len * 0.7            # short answers get partial credit
    if n > max_len:
        return max_len / n                  # overly long → penalty
    return 1.0

def _numeric_overlap(q: str, a: str) -> float:
    """Reward answers that preserve numeric facts from the question."""
    q_nums = set(_NUM_RE.findall(q))
    a_nums = set(_NUM_RE.findall(a))
    if not q_nums:
        return 0.5                          # neutral if no numbers asked
    inter = q_nums & a_nums
    return len(inter) / len(q_nums)

# -------------- public API -----------------------------------
def compute_answer_confidence(question: str, answer: str) -> float:
    """
    Parameters
    ----------
    question : original NL question
    answer   : final LLM answer string

    Returns
    -------
    float ∈ [0,1]
    """
    rel   = _relevance(question, answer)             # 0‑1
    leng  = _length_score(answer)                    # 0‑1
    numov = _numeric_overlap(question, answer)       # 0‑1

    conf = 0.6*rel + 0.2*leng + 0.2*numov
    return round(conf, 3)


# guardrails/sql_confidence.py
# --------------------------------------------------------------------
"""
Compute confidence for SQL evidence rows.

Score = 0.30·row_volume + 0.20·null_coverage + 0.50·cross_encoder
"""

# ── helpers ─────────────────────────────────────────────────────────
def _row_volume(rows: List[Tuple[Any]], k: int = 10) -> float:
    """Many answers need only a few rows; saturate at k rows."""
    return min(len(rows) / k, 1.0)

def _null_coverage(rows: List[Tuple[Any]]) -> float:
    if not rows or not isinstance(rows[0], tuple):
        return 0.0
    nulls  = sum(v in (None, "", "nan", "NaN") for r in rows for v in r)
    total  = len(rows) * len(rows[0])
    return 1 - (nulls / total) if total else 0.0

def _rows_to_text(rows: List[Tuple[Any]], max_rows: int = 6) -> str:
    """Convert first N rows to plain text for the cross-encoder."""
    return "\n".join(" | ".join(map(str, r)) for r in rows[:max_rows])

# ── public API ──────────────────────────────────────────────────────
def compute_sql_confidence(
    rows: List[Tuple[Any]],
    question: str,
    *,
    w_row: float = 0.30,
    w_cov: float = 0.20,
    w_ce:  float = 0.50,
) -> float:
    """
    Parameters
    ----------
    rows     : list[tuple]   evidence returned by `sql_db_query`
    question : str          user question

    Returns
    -------
    float ∈ [0, 1]          higher → more trustworthy answer
    """
    if not rows:
        return 0.0

    # 1) cheap heuristics
    vol_score  = _row_volume(rows)
    cov_score  = _null_coverage(rows)

    # 2) cross-encoder relevance
    cand_text  = _rows_to_text(rows)
    raw_ce     = ce_model.predict([(question, cand_text)])[0]   # 0-4 (MS-Marco)
    ce_score   = max(min(raw_ce / 4.0, 1.0), 0.0)

    # 3) weighted blend
    total_w    = w_row + w_cov + w_ce
    conf       = (w_row*vol_score + w_cov*cov_score + w_ce*ce_score) / total_w
    return round(conf, 3)

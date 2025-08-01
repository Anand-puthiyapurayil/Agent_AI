
from __future__ import annotations
import os, re
from typing import List, Tuple, Any, Sequence
import numpy as np
from sentence_transformers import CrossEncoder


_CE_NAME = os.getenv("CE_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
ce_model = CrossEncoder(_CE_NAME)           # ~110 MB, loads once

_NUM_RE = re.compile(r"-?\d+\.?\d*")

def _ce_relevance(q: str, a: str) -> float:
    raw = ce_model.predict([(q, a)])[0]          # 0-4 (MS-Marco scale)
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

def compute_rag_confidence(
    question: str,
    answer:   str,
    sim_scores: Sequence[float],                 # similarity scores from Chroma
    *,
    w_sim: float = 0.40,
    w_ce:  float = 0.30,
    w_len: float = 0.20,
    w_num: float = 0.10,
) -> float:
    if not sim_scores:
        return 0.0                               # no evidence → no trust
    best_sim  = max(sim_scores)                  # already 0-1
    ce_rel    = _ce_relevance(question, answer)
    len_rel   = _length_score(answer)
    num_rel   = _numeric_overlap(question, answer)

    tot = w_sim + w_ce + w_len + w_num
    conf = (w_sim*best_sim + w_ce*ce_rel + w_len*len_rel + w_num*num_rel) / tot
    return round(conf, 3)
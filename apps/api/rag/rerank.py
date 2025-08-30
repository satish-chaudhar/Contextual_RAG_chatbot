from __future__ import annotations
from typing import List
from dataclasses import dataclass

@dataclass
class Scored:
    text: str
    score: float
    meta: dict

def mmr_rerank(candidates: List[Scored], top_n: int = 8) -> List[Scored]:
    selected, lam = [], 0.8
    pool = candidates[:]
    if not pool: return selected
    selected.append(max(pool, key=lambda x: x.score)); pool.remove(selected[0])
    while pool and len(selected) < top_n:
        def mmr(u: Scored):
            sim_q = u.score
            div = max((abs(u.score - s.score) for s in selected), default=0.0)
            return lam * sim_q - (1 - lam) * div
        best = max(pool, key=mmr); pool.remove(best); selected.append(best)
    return selected

def cross_encoder_rerank(query: str, texts: List[Scored], top_n: int = 8) -> List[Scored]:
    try:
        from sentence_transformers import CrossEncoder
    except Exception:
        return texts[:top_n]
    model = CrossEncoder("BAAI/bge-reranker-v2-m3")
    pairs = [[query, t.text] for t in texts]
    scores = model.predict(pairs, convert_to_numpy=True).tolist()
    rescored = [Scored(text=t.text, score=s, meta=t.meta) for t, s in zip(texts, scores)]
    rescored.sort(key=lambda x: x.score, reverse=True)
    return rescored[:top_n]

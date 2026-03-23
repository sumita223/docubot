"""
rag/retriever.py — Query-time vector search
"""

import logging
from dataclasses import dataclass

import numpy as np

from cache import query_cache
from config import settings
from db import get_chunks_by_ids
from rag.ingest import get_embedder, load_faiss_index

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    text: str
    source: str
    score: float


def retrieve(query: str, top_k: int = settings.TOP_K) -> list[RetrievedChunk]:
    """
    Retrieve top-k relevant chunks for a query.
    """

    # ✅ Handle empty query
    if not query or not query.strip():
        return []

    query = query.strip()
    cache_key = query.lower()

    # ✅ Check cache
    cached = query_cache.get(cache_key)
    if cached is not None:
        logger.debug(f"Cache hit: {query}")
        return cached

    # ✅ Load FAISS index
    index, id_map = load_faiss_index()

    if index is None or index.ntotal == 0:
        logger.warning("FAISS index is empty. Run ingestion first.")
        return []

    # ✅ Embed query
    embedder = get_embedder()
    query_vec = np.array(
        embedder.encode([query], normalize_embeddings=True),
        dtype="float32"
    )

    # ✅ Search FAISS
    actual_k = min(top_k, index.ntotal)
    scores, positions = index.search(query_vec, actual_k)

    # ✅ Map FAISS positions → SQLite IDs
    sqlite_ids = []
    score_map = {}

    for pos, score in zip(positions[0], scores[0]):
        if pos < 0 or pos >= len(id_map):
            continue

        sid = id_map[int(pos)]
        sqlite_ids.append(sid)
        score_map[sid] = float(score)

    if not sqlite_ids:
        return []

    # ✅ Fetch from DB
    rows = get_chunks_by_ids(sqlite_ids)

    # ✅ Maintain correct ranking order
    id_to_row = {row["id"]: row for row in rows}

    results = []
    for sid in sqlite_ids:
        if sid in id_to_row:
            row = id_to_row[sid]
            results.append(
                RetrievedChunk(
                    text=row["text"],
                    source=row["source"],
                    score=score_map.get(sid, 0.0),
                )
            )
    # ✅ DEBUG: print retrieved chunks
    print("\n--- Retrieved Chunks ---")
    for i, r in enumerate(results):
        print(f"\nChunk {i+1} (Score: {r.score:.4f}):")
        print(r.text[:300])  # first 200 chars

    # ✅ Store in cache
    query_cache.put(cache_key, results)

    return results
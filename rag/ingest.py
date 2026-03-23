"""
rag/ingest.py — One-time document ingestion pipeline.

Flow:
    data/*.md | *.txt | *.pdf → chunker → SentenceTransformer → FAISS + SQLite

FAISS persistence:
    faiss_index.bin   — raw float32 vectors
    faiss_id_map.pkl  — mapping FAISS index → SQLite chunk ID

Called once at startup from app.py.
"""

import logging
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import fitz  # ✅ PDF support (PyMuPDF)

from config import settings
from db import insert_chunks, is_ingested, mark_ingested, get_total_chunks
from rag.chunker import chunk_text

logger = logging.getLogger(__name__)

_embedder: SentenceTransformer | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Embedding Model (Singleton)
# ─────────────────────────────────────────────────────────────────────────────

def get_embedder() -> SentenceTransformer:
    """Load the embedding model once per process."""
    global _embedder
    if _embedder is None:
        logger.info(f"Loading embedding model: {settings.EMBED_MODEL}")
        _embedder = SentenceTransformer(settings.EMBED_MODEL)
    return _embedder


# ─────────────────────────────────────────────────────────────────────────────
# PDF TEXT EXTRACTION (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def extract_pdf_text(path: Path, max_pages: int = 40) -> str:
    """
    Extract text from PDF (limited pages for performance).
    """
    doc = fitz.open(path)
    text = []

    for i, page in enumerate(doc):
        if i >= max_pages:
            break
        text.append(page.get_text())

    return " ".join(text)


# ─────────────────────────────────────────────────────────────────────────────
# FAISS helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_faiss_index() -> tuple[faiss.Index | None, list[int]]:
    if (
        Path(settings.FAISS_INDEX_PATH).exists()
        and Path(settings.FAISS_ID_MAP_PATH).exists()
    ):
        index = faiss.read_index(settings.FAISS_INDEX_PATH)
        with open(settings.FAISS_ID_MAP_PATH, "rb") as f:
            id_map: list[int] = pickle.load(f)
        logger.info(f"Loaded FAISS index — {index.ntotal} vectors.")
        return index, id_map
    return None, []


def save_faiss_index(index: faiss.Index, id_map: list[int]) -> None:
    # ✅ Ensure directory exists
    Path(settings.FAISS_INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, settings.FAISS_INDEX_PATH)

    with open(settings.FAISS_ID_MAP_PATH, "wb") as f:
        pickle.dump(id_map, f)

    logger.info(f"Saved FAISS index — {index.ntotal} vectors.")


def _get_or_create_index(dim: int) -> tuple[faiss.Index, list[int]]:
    index, id_map = load_faiss_index()
    if index is None:
        index = faiss.IndexFlatIP(dim)
        id_map = []
    return index, id_map


# ─────────────────────────────────────────────────────────────────────────────
# MAIN INGESTION
# ─────────────────────────────────────────────────────────────────────────────

def ingest_documents(data_dir: str = "data/") -> None:
    """
    Scan data_dir for .md, .txt, .pdf and ingest new files only.
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        logger.warning(f"'{data_dir}' not found. Create it and add files.")
        return

    # ✅ Added PDF support
    all_files = (
        sorted(data_path.glob("*.md")) +
        sorted(data_path.glob("*.txt")) +
        sorted(data_path.glob("*.pdf"))
    )

    new_files = [f for f in all_files if not is_ingested(f.name)]

    if not new_files:
        logger.info(f"All docs ingested — {get_total_chunks()} chunks in DB.")
        return

    embedder = get_embedder()
    dim = embedder.encode(["warmup"], normalize_embeddings=True).shape[1]

    index, id_map = _get_or_create_index(dim)

    for filepath in new_files:
        logger.info(f"Ingesting: {filepath.name}")

        # ✅ Handle different file types
        if filepath.suffix == ".pdf":
            raw_text = extract_pdf_text(filepath)
        else:
            raw_text = filepath.read_text(encoding="utf-8")

        # ✅ Clean text
        raw_text = raw_text.replace("\n", " ").strip()

        if not raw_text:
            logger.warning(f"Empty file: {filepath.name} — skipping.")
            continue

        chunks = chunk_text(raw_text)

        if not chunks:
            logger.warning(f"No chunks from {filepath.name} — skipping.")
            continue

        embeddings = embedder.encode(
            chunks,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        embeddings = np.array(embeddings, dtype="float32")

        db_ids = insert_chunks(source=filepath.name, chunks=chunks)

        index.add(embeddings)
        id_map.extend(db_ids)

        mark_ingested(filepath.name)

        logger.info(f"  ✓ {len(chunks)} chunks from {filepath.name}")

    save_faiss_index(index, id_map)

    logger.info("✅ Ingestion complete.")
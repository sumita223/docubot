"""
rag/chunker.py — Semantic-aware text splitter using sentence grouping.

Improves:
- Keeps full sentences intact
- Maintains semantic meaning
- Avoids broken chunks
"""

import re
from config import settings


def split_into_sentences(text: str) -> list[str]:
    """
    Basic sentence splitter using regex.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(
    text: str,
    chunk_size: int = settings.CHUNK_SIZE,
    overlap: int = settings.CHUNK_OVERLAP,
) -> list[str]:

    if not text.strip():
        return []

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    sentences = split_into_sentences(text)

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())

        # if adding this sentence exceeds chunk size → finalize chunk
        if current_length + sentence_length > chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk).strip())

            # overlap handling (keep last few words)
            if overlap > 0 and current_chunk:
                overlap_words = " ".join(current_chunk).split()[-overlap:]
                current_chunk = [" ".join(overlap_words)]
                current_length = len(overlap_words)
            else:
                current_chunk = []
                current_length = 0

        current_chunk.append(sentence)
        current_length += sentence_length

    # add last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())

    return chunks
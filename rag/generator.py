# """
# rag/generator.py — Build RAG prompt and generate answer using OpenAI.
# """

# import logging
# from typing import List, Tuple

# from openai import OpenAI

# from config import settings
# from rag.retriever import RetrievedChunk

# logger = logging.getLogger(__name__)

# client = OpenAI(api_key=settings.OPENAI_API_KEY)


# # ─────────────────────────────────────────────────────────────────────────────
# # SYSTEM PROMPT
# # ─────────────────────────────────────────────────────────────────────────────

# SYSTEM_PROMPT = """You are DocuBot, a helpful assistant.

# Use the provided context to answer the question.

# Even if the information is partial or not perfectly structured,
# try to answer using the available context.

# Only say "I don't have that information in my knowledge base"
# if the context is completely unrelated to the question.

# Do not hallucinate facts. Keep answers clear and concise.
# """


# # ─────────────────────────────────────────────────────────────────────────────
# # BUILD PROMPT
# # ─────────────────────────────────────────────────────────────────────────────

# def build_prompt(query: str, chunks: List[RetrievedChunk]) -> str:
#     context_parts = []
#     total_chars = 0

#     for chunk in chunks:
#         entry = f"[Source: {chunk.source}]\n{chunk.text}"

#         if total_chars + len(entry) > settings.MAX_CONTEXT_CHARS:
#             break

#         context_parts.append(entry)
#         total_chars += len(entry)

#     context_block = (
#         "\n\n---\n\n".join(context_parts)
#         if context_parts
#         else "No relevant context found."
#     )

#     return f"""
# Context:
# {context_block}

# Question: {query}
# """


# # ─────────────────────────────────────────────────────────────────────────────
# # GENERATE ANSWER
# # ─────────────────────────────────────────────────────────────────────────────

# def generate_answer(
#     query: str,
#     chunks: List[RetrievedChunk]
# ) -> Tuple[str, List[str]]:

#     # ✅ Empty query check
#     if not query or not query.strip():
#         return "Please provide a valid question.", []

#     query = query.strip()

#     # ✅ No chunks
#     if not chunks:
#         return "I don't have that information in my knowledge base.", []

#     # ✅ Sort by relevance
#     chunks = sorted(chunks, key=lambda x: x.score, reverse=True)

#     # ✅ Threshold from config (NOT hardcoded)
#     best_score = chunks[0].score
#     min_score = getattr(settings, "MIN_RELEVANCE_SCORE", 0.0)

#     if best_score < min_score:
#         logger.info(f"Low relevance score: {best_score:.4f}")
#         return "I don't have that information in my knowledge base.", []

#     # ✅ Source filtering (config-driven)
#     excluded_keywords = getattr(settings, "EXCLUDED_SOURCES", [])

#     filtered_chunks = [
#         chunk for chunk in chunks
#         if not any(keyword.lower() in chunk.source.lower()
#                    for keyword in excluded_keywords)
#     ]

#     # fallback if everything removed
#     if not filtered_chunks:
#         filtered_chunks = chunks

#     # ✅ Build prompt
#     prompt = build_prompt(query, filtered_chunks)

#     try:
#         response = client.chat.completions.create(
#             model=settings.OPENAI_MODEL,
#             messages=[
#                 {"role": "system", "content": SYSTEM_PROMPT},
#                 {"role": "user", "content": prompt},
#             ],
#             temperature=0.2,
#             max_tokens=512,
#         )

#         answer = response.choices[0].message.content.strip()

#     except Exception as e:
#         logger.error(f"LLM error: {e}")
#         return "Error generating response. Please try again.", []

#     # ✅ Extract sources from filtered chunks
#     sources = sorted({chunk.source for chunk in filtered_chunks})

#     return answer, sources


"""
rag/generator.py — Build RAG prompt and generate answer using OpenAI
"""

import logging
from typing import List, Tuple

from openai import OpenAI

from config import settings
from rag.retriever import RetrievedChunk
from bot.history import get_user_history, add_user_message

logger = logging.getLogger(__name__)

client = OpenAI(api_key=settings.OPENAI_API_KEY)


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are DocuBot, a helpful assistant.

Answer the question using ONLY the provided context.

If the context contains partial or related information, use it to answer as best as possible.

Only say "I don't have that information in my knowledge base"
if the context is completely unrelated.

Keep answers clear, structured, and concise.
"""


# ─────────────────────────────────────────────────────────────────────────────
# BUILD CONTEXT
# ─────────────────────────────────────────────────────────────────────────────

def build_context(chunks: List[RetrievedChunk]) -> str:
    context_parts = []
    total_chars = 0

    for chunk in chunks:
        entry = f"[Source: {chunk.source}]\n{chunk.text}"

        if total_chars + len(entry) > settings.MAX_CONTEXT_CHARS:
            break

        context_parts.append(entry)
        total_chars += len(entry)

    return (
        "\n\n---\n\n".join(context_parts)
        if context_parts
        else "No relevant context found."
    )


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE ANSWER
# ─────────────────────────────────────────────────────────────────────────────

def generate_answer(
    user_id: int,   # ✅ REQUIRED
    query: str,
    chunks: List[RetrievedChunk]
) -> Tuple[str, List[str]]:

    # ✅ Safety checks
    if not query or not query.strip():
        return "Please provide a valid question.", []

    if not chunks:
        return "I don't have that information in my knowledge base.", []

    query = query.strip()

    # ✅ Sort by relevance
    chunks = sorted(chunks, key=lambda x: x.score, reverse=True)

    # ✅ Relevance threshold
    best_score = chunks[0].score
    min_score = getattr(settings, "MIN_RELEVANCE_SCORE", 0.0)

    if best_score < min_score:
        logger.info(f"Low relevance score: {best_score:.4f}")
        return "I don't have that information in my knowledge base.", []

    # ✅ Build context
    context = build_context(chunks)

    user_prompt = f"""
Context:
{context}

Question: {query}
"""

    # ✅ GET HISTORY (ACTUAL USAGE)
    history = get_user_history(user_id)

    # ✅ BUILD MESSAGE LIST
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=400,
        )

        answer = response.choices[0].message.content.strip()

    except Exception as e:
        logger.error(f"LLM error: {e}")
        return "Error generating response. Please try again.", []

    # ✅ STORE HISTORY (CRITICAL STEP)
    

    # ✅ Extract sources
    sources = sorted({chunk.source for chunk in chunks})

    return answer, sources
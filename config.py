"""
config.py — Simple config for DocuBot (assignment version)
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def _get_int(name: str, default: str) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return int(default)


@dataclass(frozen=True)
class Settings:
    # Telegram
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")

    # OpenAI (single LLM as per assignment)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Embedding
    EMBED_MODEL: str = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
    CACHE_MAX_SIZE: int = 100
    MAX_CONTEXT_CHARS: int = 3000

    # Paths
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "data/faiss_index.bin")
    FAISS_ID_MAP_PATH = "data/faiss_id_map.pkl"
    DB_PATH: str = os.getenv("DB_PATH", "data/docubot.db")

    # RAG tuning
    CHUNK_SIZE: int = _get_int("CHUNK_SIZE", "120")
    CHUNK_OVERLAP: int = _get_int("CHUNK_OVERLAP", "30")
    TOP_K: int = _get_int("TOP_K", "5")

    def validate(self) -> None:
        if not self.TELEGRAM_BOT_TOKEN:
            raise ValueError("TELEGRAM_BOT_TOKEN missing in .env")

        if not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY missing in .env")


settings = Settings()
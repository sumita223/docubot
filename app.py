"""
app.py — DocuBot entry point (Assignment Version)

Startup sequence:
    1. Validate config
    2. Initialise DB
    3. Run ingestion (if needed)
    4. Start Telegram bot
"""

import logging
import os

from telegram.ext import Application, CommandHandler

from config import settings
from db import init_db
from rag.ingest import ingest_documents
from bot.handlers import ask_handler, help_handler, start_handler, clear_handler, history_handler, summarize_handler

# Logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main() -> None:
    # ✅ Validate config
    settings.validate()

    # ✅ Ensure folders exist
    os.makedirs("data", exist_ok=True)

    # ✅ Init DB
    logger.info("Initialising database...")
    init_db()

    # ✅ Run ingestion only if index not present
    if not os.path.exists(settings.FAISS_INDEX_PATH):
        logger.info("Running ingestion...")
        ingest_documents(data_dir="data/")
    else:
        logger.info("FAISS index already exists. Skipping ingestion.")

    # ✅ Start bot
    app = Application.builder().token(settings.TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("help", help_handler))
    app.add_handler(CommandHandler("ask", ask_handler))
    app.add_handler(CommandHandler("clear", clear_handler))
    app.add_handler(CommandHandler("history", history_handler))
    app.add_handler(CommandHandler("summarize", summarize_handler))
    

    logger.info("🚀 DocuBot is live...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()

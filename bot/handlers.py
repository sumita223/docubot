"""
bot/handlers.py — Telegram command handlers
"""

from telegram import Update
from telegram.ext import ContextTypes
from bot.history import add_user_message, get_user_history, clear_user_history
from rag.retriever import retrieve
from rag.generator import generate_answer


# ─────────────────────────────────────────────────────────────────────────────
# /start
# ─────────────────────────────────────────────────────────────────────────────

async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Welcome to DocuBot!\n\nUse /ask <question> to query documents."
    )


# ─────────────────────────────────────────────────────────────────────────────
# /help
# ─────────────────────────────────────────────────────────────────────────────

async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📌 Commands:\n"
        "/ask <query> — Ask questions from documents\n"
        "/help — Show this message"
    )


# ─────────────────────────────────────────────────────────────────────────────
# /ask
# ─────────────────────────────────────────────────────────────────────────────

async def ask_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = " ".join(context.args)

    if not query:
        await update.message.reply_text("Please provide a question.\nExample: /ask What is AI?")
        return

    # Step 1: Retrieve relevant chunks
    # chunks = retrieve(query)

    # # Step 2: Generate answer
    # answer, sources = generate_answer(query, chunks)

    user_id = update.effective_user.id
    add_user_message(user_id, "user", query)

    # ✅ get previous history
    history = get_user_history(user_id)
    print("\n--- HISTORY DEBUG ---")
    for i, msg in enumerate(history):
        print(f"{i+1}. {msg}")

    # retrieve
    chunks = retrieve(query)

    # generate (pass history)
    answer, sources = generate_answer(user_id, query, chunks)

    # ✅ store new interaction
    
    add_user_message(user_id, "assistant", answer)

    # Step 3: Format response
    if sources:
        source_text = "\n".join(f"- {s}" for s in sources)
        reply = f"📌 {answer}\n\n📄 Sources:\n{source_text}"
    else:
        reply = f"📌 {answer}"

    # Step 4: Send response
    await update.message.reply_text(reply)

async def clear_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    clear_user_history(user_id)
    await update.message.reply_text("🧹 History cleared!")

async def history_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    history = get_user_history(user_id)

    if not history:
        await update.message.reply_text("No history yet.")
        return

    text = "\n\n".join([f"{msg['role']}: {msg['content']}" for msg in history])

    await update.message.reply_text(f"📜 History:\n\n{text}")

async def summarize_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    history = get_user_history(user_id)

    if not history:
        await update.message.reply_text("No conversation to summarize.")
        return

    # convert history to text
    transcript = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])

    from rag.generator import generate_summary
    summary = generate_summary(transcript)

    await update.message.reply_text(f"📝 Summary:\n{summary}")
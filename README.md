# DocuBot 🤖

A Telegram-based Retrieval-Augmented Generation (RAG) chatbot that answers questions from your documents using FAISS vector search, SQLite metadata storage, and OpenAI's GPT-4o-mini.

---

## Features

- `/ask <question>` — Query your documents using RAG
- `/start` — Welcome message
- `/help` — List of available commands
- `/clear` — Clear conversation history
- `/history` — View past queries
- `/summarize` — Summarise ingested documents
- LRU caching of repeated queries for fast responses
- Incremental ingestion (skips already-processed files)

---

## Optional Enhancements Implemented

-  Message history awareness (last 3 interactions per user using deque)
-  LRU caching to avoid repeated embedding and LLM calls
-  Source attribution in responses
-  /summarize command for conversation summarization

---

## Key Design Decisions

- Used FAISS for efficient vector similarity search
- Used SQLite to map vector indices to original text chunks
- Implemented hybrid retrieval (semantic + keyword filtering) for better accuracy
- Used a fixed-size deque for lightweight conversational memory
- Limited context size to reduce latency and avoid token overflow

---

## Project Structure

```

docubot/
├── app.py                  # Entry point — starts bot, triggers ingestion
├── db.py                   # SQLite setup (chunks table + ingestion log)
├── cache.py                # In-memory LRU query cache
├── config.py                
├── bot/
│   ├── handlers.py         # /start /help /ask /history /summarize
│   └── history.py          # Per-user last-3-turns memory
├── rag/
│   ├── ingest.py           # Doc loading → chunking → embedding → FAISS
│   ├── retriever.py        # Query-time FAISS search → SQLite chunk lookup
│   ├── chunker.py        
│   └── generator.py        # Prompt builder + OpenAI/Ollama LLM call
├── data/                   # Your knowledge base (.md or .txt files)
│   ├── artificial_intelligence_tutorial.pdf
│   ├── Easy_recipes.pdf
│   ├── Oracle_User_Guide.pdf
├── .env.example            # Copy to .env and fill in tokens
└── requirements.txt


---

## How to Run Locally

### Prerequisites

- Python 3.10+
- A Telegram bot token (from [@BotFather](https://t.me/BotFather))
- An OpenAI API key

### 1. Clone and install dependencies

```bash
git clone <your-repo-url>
cd docubot
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o-mini          # optional, default: gpt-4o-mini
EMBED_MODEL=all-MiniLM-L6-v2      # optional, default: all-MiniLM-L6-v2
FAISS_INDEX_PATH=data/faiss_index.bin
DB_PATH=data/docubot.db
CHUNK_SIZE=120
CHUNK_OVERLAP=30
TOP_K=5
```

### 3. Add your documents

Place `.txt`, `.pdf`, or other supported files inside the `data/` directory.

### 4. Start the bot

```bash
python app.py
```

On first run, the bot will:
1. Validate your config
2. Initialise the SQLite database
3. Chunk and embed all documents into the FAISS index
4. Start polling for Telegram messages

Subsequent runs skip ingestion if the FAISS index already exists.

---

## Running with Docker Compose

### `Dockerfile`

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```


### Run

```bash
docker compose up --build
```

The `data/` volume is mounted so the FAISS index and SQLite database persist across container restarts.

---

## Models and APIs Used

| Component | Model / Library | Purpose |
|---|---|---|
| LLM | `gpt-4o-mini` (OpenAI) | Answer generation from retrieved context |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) | Encode text chunks into vectors |
| Vector store | FAISS (Meta) | Approximate nearest-neighbour search |
| Metadata store | SQLite | Map FAISS positions → chunk text + source |
| Bot framework | python-telegram-bot | Telegram bot interface |
| Caching | In-memory LRU (custom) | Avoid re-calling OpenAI for identical queries |

---

## System Design

```
User (Telegram)
      │
      ▼
 Telegram Bot (python-telegram-bot)
      │
      ├─ /ask <query>
      │       │
      │  LRU Cache hit? ──► Return cached answer
      │       │ miss
      │       ▼
      │  Embed query (all-MiniLM-L6-v2)
      │       │
      │       ▼
      │  FAISS index ──► Top-K vector IDs
      │       │
      │       ▼
      │  SQLite (chunks table) ──► Chunk texts + sources
      │       │
      │       ▼
      │  Build prompt (context + question)
      │       │
      │       ▼
      │  OpenAI gpt-4o-mini ──► Answer
      │       │
      │  Store in LRU cache
      │       │
      │       ▼
      │  Reply to user
      │
      └─ /history, /clear, /summarize → In-memory history(deque)
```

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | — | **Required.** From @BotFather |
| `OPENAI_API_KEY` | — | **Required.** From platform.openai.com |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model for generation |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model for embeddings |
| `CHUNK_SIZE` | `120` | Token size per text chunk |
| `CHUNK_OVERLAP` | `30` | Overlap between consecutive chunks |
| `TOP_K` | `5` | Number of chunks retrieved per query |
| `FAISS_INDEX_PATH` | `data/faiss_index.bin` | Path to persisted FAISS index |
| `DB_PATH` | `data/docubot.db` | Path to SQLite database |
| `CACHE_MAX_SIZE` | `100` | Maximum LRU cache entries |
| `MAX_CONTEXT_CHARS` | `300` | Max characters of context sent to the LLM |

---

## Notes

- The bot uses **polling** mode (not webhooks). For production, consider switching to webhook mode behind a reverse proxy.
- Re-ingestion is skipped on restart via the `ingestion_log` table in SQLite. To force re-ingestion, delete `data/faiss_index.bin` and restart.
- The LRU cache is in-memory only and resets on restart.
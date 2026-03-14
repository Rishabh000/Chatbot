# Real Estate Chatbot

A RAG-powered chatbot that answers real estate questions using Wikipedia as its knowledge base. Built with FastAPI, LangChain, ChromaDB, and Gemini.

## Demo: https://real-estatechatbot.vercel.app/

## Features

- **Wikipedia-backed knowledge** — Fetches and indexes 28 real estate Wikipedia pages (with source attribution) into a vector store
- **Guardrails** — LLM-based topic classifier blocks non-real-estate questions
- **Response verification** — Second LLM pass confirms answers are grounded in context
- **In-memory cache** — LRU cache avoids redundant LLM calls for repeated questions
- **Chat persistence** — All messages stored in SQLite via async SQLAlchemy
- **Rate limiting** — Per-IP rate limiting via SlowAPI

## Project Structure

```
Chatbot/
├── app/
│   ├── main.py             # API Gateway (routing, rate limiting)
│   ├── pipeline.py          # Orchestrates the backend pod flow
│   ├── guardrails.py        # Guardrail Classifier
│   ├── retriever.py         # Retriever (ChromaDB similarity search)
│   ├── prompt_builder.py    # Prompt Builder (RAG prompt assembly)
│   ├── llm_client.py        # LLM Client (Gemini wrapper)
│   ├── verifier.py          # Verifier (response grounding check)
│   ├── cache.py             # In-memory LRU response cache
│   ├── vector_store.py      # Vector DB (ChromaDB + Gemini embeddings)
│   ├── wikipedia_kb.py      # Wikipedia Knowledge Base (fetch + chunk)
│   ├── database.py          # Chat Storage (async SQLAlchemy)
│   ├── models.py            # ChatMessage ORM model
│   ├── schemas.py           # Request/response Pydantic models
│   └── config.py            # Pydantic settings (env-driven)
├── static/
│   └── index.html           # Chat UI
├── api/
│   └── index.py             # Vercel entry point
├── build_knowledge_base.py  # Pre-build vector store script
├── vercel.json
├── requirements.txt
├── .env.example
└── .gitignore
```

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/Rishabh000/Chatbot.git
cd Chatbot
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your Gemini API key
```

### 3. Run the server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

On first launch the server will:
1. Create the SQLite database (`chatbot.db`)
2. Fetch ~28 Wikipedia pages on real estate topics, embed them with Gemini embeddings, and build the ChromaDB vector store (`chroma_db/`)

Subsequent launches skip the Wikipedia fetch if the vector store already exists.

### 4. Try it out

Open **http://localhost:8000** for the chat UI, or use curl:

```bash
curl -X POST http://localhost:8000/session/new

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "YOUR_SESSION_ID", "question": "What is a mortgage?"}'
```

Interactive API docs at **http://localhost:8000/docs**.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/session/new` | Generate a new session ID |
| `POST` | `/chat` | Send a question (body: `session_id`, `question`) |
| `GET` | `/chat/{session_id}/history` | Retrieve chat history for a session |

## How It Works

### Request Pipeline

```
Cache check → Guardrail → Retriever → Prompt Builder → LLM → Verifier → Cache set
```

1. **Cache** — Check if this question was answered recently (LRU, 256 entries).
2. **Guardrail** — Gemini classifies the question as ALLOWED or BLOCKED.
3. **Retrieve** — Top 5 relevant chunks from ChromaDB via similarity search.
4. **Prompt Builder** — Assembles context + question into a RAG prompt.
5. **LLM** — Gemini generates an answer grounded in the retrieved context.
6. **Verifier** — Second Gemini call confirms the answer isn't hallucinated.

### Database Schema

| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer (PK) | Auto-increment ID |
| `session_id` | String(64) | Groups messages into conversations |
| `role` | String(16) | `"user"` or `"assistant"` |
| `content` | Text | The message body |
| `was_blocked` | Boolean | `True` if the guardrail blocked this response |
| `created_at` | DateTime | Timestamp |

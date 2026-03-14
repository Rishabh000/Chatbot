"""API Gateway — routing, rate limiting, static serving. No business logic."""

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import IS_VERCEL, get_settings
from app.database import get_db, init_db
from app.models import ChatMessage
from app.pipeline import run as run_pipeline
from app.schemas import ChatHistoryItem, ChatHistoryResponse, ChatRequest, ChatResponse
from app.vector_store import build_store
from app.wikipedia_kb import fetch_and_chunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if IS_VERCEL:
        logger.info("Running on Vercel — skipping startup build (using bundled store).")
    else:
        await init_db()
        logger.info("Database initialized.")
        chunks = fetch_and_chunk()
        build_store(chunks)
        logger.info("Vector store ready.")
    yield


app = FastAPI(
    title="Real Estate Chatbot",
    version="1.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
@limiter.limit(settings.rate_limit)
async def chat(request: Request, body: ChatRequest):
    session_id = body.session_id or str(uuid.uuid4())

    if not IS_VERCEL:
        async for db in get_db():
            db.add(ChatMessage(session_id=session_id, role="user", content=body.question))
            await db.commit()

    answer, was_blocked = await run_pipeline(body.question)

    if not IS_VERCEL:
        async for db in get_db():
            db.add(ChatMessage(
                session_id=session_id, role="assistant", content=answer, was_blocked=was_blocked
            ))
            await db.commit()

    return ChatResponse(session_id=session_id, answer=answer, was_blocked=was_blocked)


@app.get("/chat/{session_id}/history", response_model=ChatHistoryResponse)
@limiter.limit(settings.rate_limit)
async def chat_history(request: Request, session_id: str):
    if IS_VERCEL:
        return ChatHistoryResponse(session_id=session_id, messages=[])

    async for db in get_db():
        result = await db.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at)
        )
        return ChatHistoryResponse(
            session_id=session_id,
            messages=[
                ChatHistoryItem(
                    role=m.role, content=m.content,
                    was_blocked=m.was_blocked, created_at=m.created_at,
                )
                for m in result.scalars().all()
            ],
        )


@app.post("/session/new")
async def new_session():
    return {"session_id": str(uuid.uuid4())}


STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
if STATIC_DIR.is_dir():
    @app.get("/")
    async def serve_ui():
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

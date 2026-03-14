import logging
import os
import shutil
import time

from google import genai
from google.genai.errors import ClientError
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import Chroma

from app.config import IS_VERCEL, get_settings

BUNDLED_CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")

logger = logging.getLogger(__name__)

BATCH_SIZE = 25
MAX_RETRIES = 6
BATCH_DELAY = 3


class GeminiEmbeddings(Embeddings):
    """LangChain-compatible wrapper around the google-genai embedding client."""

    def __init__(self, api_key: str, model: str = "gemini-embedding-001"):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def _call_with_retry(self, contents):
        for attempt in range(MAX_RETRIES):
            try:
                return self.client.models.embed_content(
                    model=self.model, contents=contents
                )
            except ClientError as e:
                if "429" in str(e) and attempt < MAX_RETRIES - 1:
                    wait = min(2 ** attempt * 15, 120)
                    logger.warning("Rate limited, retrying in %ds... (%d/%d)", wait, attempt + 1, MAX_RETRIES)
                    time.sleep(wait)
                else:
                    raise

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        all_embeddings: list[list[float]] = []
        total = len(texts)
        num_batches = -(-total // BATCH_SIZE)
        for i in range(0, total, BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            logger.info("Embedding batch %d/%d", i // BATCH_SIZE + 1, num_batches)
            result = self._call_with_retry(batch)
            all_embeddings.extend(e.values for e in result.embeddings)
            if i + BATCH_SIZE < total:
                time.sleep(BATCH_DELAY)
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        result = self._call_with_retry(text)
        return result.embeddings[0].values


def _get_embeddings() -> GeminiEmbeddings:
    settings = get_settings()
    return GeminiEmbeddings(api_key=settings.gemini_api_key)


def _ensure_vercel_store():
    """On Vercel, copy the bundled chroma_db to /tmp if not already there."""
    settings = get_settings()
    target = settings.chroma_persist_dir
    if os.path.exists(target):
        return
    if os.path.exists(BUNDLED_CHROMA_DIR):
        logger.info("Copying bundled vector store to %s ...", target)
        shutil.copytree(BUNDLED_CHROMA_DIR, target)
    else:
        raise RuntimeError(
            "No bundled chroma_db found. Run 'python build_knowledge_base.py' locally first."
        )


def get_store() -> Chroma:
    settings = get_settings()
    if IS_VERCEL:
        _ensure_vercel_store()
    return Chroma(
        persist_directory=settings.chroma_persist_dir,
        embedding_function=_get_embeddings(),
        collection_name="real_estate",
    )


def build_store(chunks: list) -> Chroma:
    """Build a new vector store from document chunks. Skips if already populated."""
    settings = get_settings()
    embeddings = _get_embeddings()

    try:
        store = Chroma(
            persist_directory=settings.chroma_persist_dir,
            embedding_function=embeddings,
            collection_name="real_estate",
        )
        if store._collection.count() > 0:
            logger.info(
                "Vector store already has %d chunks, skipping rebuild.",
                store._collection.count(),
            )
            return store
    except Exception:
        pass

    logger.info("Indexing %d chunks into ChromaDB...", len(chunks))
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=settings.chroma_persist_dir,
        collection_name="real_estate",
    )
    logger.info("Vector store persisted to %s", settings.chroma_persist_dir)
    return store

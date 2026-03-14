import os
from functools import lru_cache

from pydantic_settings import BaseSettings

IS_VERCEL = bool(os.environ.get("VERCEL"))


class Settings(BaseSettings):
    gemini_api_key: str = ""
    database_url: str = "sqlite+aiosqlite:///./chatbot.db"
    rate_limit: str = "20/minute"
    chroma_persist_dir: str = "./chroma_db"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    if IS_VERCEL:
        settings.chroma_persist_dir = "/tmp/chroma_db"
        settings.database_url = "sqlite+aiosqlite:////tmp/chatbot.db"
    return settings

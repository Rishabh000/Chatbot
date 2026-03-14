from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import get_settings


def get_llm(temperature: float = 0.3) -> ChatGoogleGenerativeAI:
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=temperature,
        google_api_key=settings.gemini_api_key,
    )

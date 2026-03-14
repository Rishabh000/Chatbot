"""
Orchestrates the backend pod flow:

  Cache check → Guardrail → Retriever → Prompt Builder → LLM Client → Verifier → Cache set
"""

import logging

from google.api_core.exceptions import ResourceExhausted

from app.cache import response_cache
from app.guardrails import BLOCK_MESSAGE, is_real_estate_question
from app.llm_client import get_llm
from app.prompt_builder import build
from app.retriever import retrieve
from app.verifier import verify

logger = logging.getLogger(__name__)

NO_RESULTS = (
    "I couldn't find relevant real estate information to answer your question. "
    "Could you rephrase or ask about a different real estate topic?"
)

RATE_LIMITED = (
    "I'm currently experiencing high demand. Please wait a minute and try again."
)


async def run(question: str) -> tuple[str, bool]:
    """
    Execute the full chatbot pipeline.

    Returns (answer, was_blocked).
    """
    cached = response_cache.get(question)
    if cached:
        return cached, False

    try:
        if not await is_real_estate_question(question):
            return BLOCK_MESSAGE, True

        docs = retrieve(question)
        if not docs:
            return NO_RESULTS, False

        payload = build(question, docs)
        llm = get_llm()
        chain = payload["prompt"] | llm
        raw_answer = await chain.ainvoke(payload["variables"])
        answer_text = raw_answer.content

        answer_text = await verify(
            question=question,
            context=payload["variables"]["context"],
            answer=answer_text,
        )

        response_cache.set(question, answer_text)
        return answer_text, False

    except ResourceExhausted as e:
        logger.warning("Gemini rate limit hit: %s", e)
        return RATE_LIMITED, False

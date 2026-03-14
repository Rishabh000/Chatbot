from langchain.prompts import ChatPromptTemplate

from app.llm_client import get_llm

VERIFY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a response verifier. Check whether the ANSWER is grounded "
                "in the provided CONTEXT and is relevant to the QUESTION.\n\n"
                "Respond with EXACTLY one word:\n"
                "  PASS  — answer is grounded and relevant\n"
                "  FAIL  — answer is hallucinated, off-topic, or contradicts the context"
            ),
        ),
        (
            "human",
            "QUESTION: {question}\n\nCONTEXT: {context}\n\nANSWER: {answer}",
        ),
    ]
)

FALLBACK = (
    "I found some information but couldn't produce a reliable answer. "
    "Could you rephrase your real estate question?"
)


async def verify(question: str, context: str, answer: str) -> str:
    """Return the original answer if it passes verification, otherwise a fallback."""
    llm = get_llm(temperature=0)
    chain = VERIFY_PROMPT | llm
    result = await chain.ainvoke(
        {"question": question, "context": context, "answer": answer}
    )
    if result.content.strip().upper() == "PASS":
        return answer
    return FALLBACK

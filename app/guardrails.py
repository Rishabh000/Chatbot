from langchain.prompts import ChatPromptTemplate

from app.llm_client import get_llm

TOPIC_CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a strict topic classifier. Your ONLY job is to decide whether "
                "a user's question is related to real estate. Real estate includes: "
                "buying, selling, renting, leasing property; mortgages and home loans; "
                "property valuation and appraisal; zoning and land use; property taxes; "
                "real estate investment (REITs, flipping, rental income); property management; "
                "commercial and residential real estate; housing markets and trends; "
                "construction and development; home inspection; title and escrow; "
                "tenant and landlord law; HOAs and condos.\n\n"
                "Respond with EXACTLY one word:\n"
                "  ALLOWED  — if the question is about real estate\n"
                "  BLOCKED  — if the question is NOT about real estate"
            ),
        ),
        ("human", "{question}"),
    ]
)

BLOCK_MESSAGE = (
    "I'm sorry, but I can only answer questions related to real estate. "
    "Please ask me about topics like buying or selling property, mortgages, "
    "property management, real estate investment, zoning, or housing markets."
)


async def is_real_estate_question(question: str) -> bool:
    llm = get_llm(temperature=0)
    chain = TOPIC_CLASSIFIER_PROMPT | llm
    response = await chain.ainvoke({"question": question})
    return response.content.strip().upper() == "ALLOWED"

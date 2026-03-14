from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document

SYSTEM_PROMPT = (
    "You are a knowledgeable real estate expert. Answer the user's question "
    "using ONLY the provided Wikipedia context below. Follow these rules:\n"
    "- Cite the Wikipedia source article when referencing specific concepts "
    "(e.g. \"According to the Wikipedia article on Mortgage loan, ...\").\n"
    "- Structure your answer clearly with paragraphs. Use **bold** for key terms.\n"
    "- If the context does not contain enough information, say so honestly "
    "rather than making things up.\n\n"
    "Context:\n{context}"
)

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


def build(question: str, docs: list[Document]) -> dict:
    chunks: list[str] = []
    for doc in docs:
        source = doc.metadata.get("source", "Wikipedia")
        chunks.append(f"[{source}]\n{doc.page_content}")
    context = "\n\n---\n\n".join(chunks)
    return {"prompt": QA_PROMPT, "variables": {"context": context, "question": question}}

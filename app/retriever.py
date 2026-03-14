from langchain.schema import Document

from app.vector_store import get_store


def retrieve(question: str, k: int = 5) -> list[Document]:
    store = get_store()
    return store.as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    ).invoke(question)

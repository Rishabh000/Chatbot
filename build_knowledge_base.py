"""
Pre-build the ChromaDB vector store from Wikipedia.

    python build_knowledge_base.py
"""

from dotenv import load_dotenv

load_dotenv()

from app.wikipedia_kb import fetch_and_chunk  
from app.vector_store import build_store 

if __name__ == "__main__":
    chunks = fetch_and_chunk()
    store = build_store(chunks)
    print(f"Vector store ready — {store._collection.count()} chunks in ./chroma_db")

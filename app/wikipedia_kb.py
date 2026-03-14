import logging

import requests
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

REAL_ESTATE_TOPICS = [
    "Real estate",
    "Mortgage loan",
    "Real estate investing",
    "Property management",
    "Zoning",
    "Foreclosure",
    "Real estate investment trust",
    "Real estate appraisal",
    "Property tax",
    "Lease",
    "Real estate broker",
    "Home inspection",
    "Title insurance",
    "Escrow",
    "Homeowner association",
    "Condominium",
    "Commercial property",
    "Real estate economics",
    "Housing bubble",
    "Eminent domain",
    "Land use",
    "Rental agreement",
    "Real estate development",
    "Closing (real estate)",
    "Down payment",
    "Equity (finance)",
    "Refinancing",
    "Housing affordability index",
]

WIKI_API = "https://en.wikipedia.org/w/api.php"


def _fetch_page(title: str) -> tuple[str, str] | None:
    """Fetch a Wikipedia page's full text via the MediaWiki API."""
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
        "format": "json",
    }
    try:
        resp = requests.get(WIKI_API, params=params, timeout=15)
        resp.raise_for_status()
        pages = resp.json()["query"]["pages"]
        page = next(iter(pages.values()))
        if "extract" not in page:
            return None
        return page["title"], page["extract"]
    except Exception as e:
        logger.warning("Failed to fetch '%s': %s", title, e)
        return None


def fetch_and_chunk() -> list[Document]:
    """Fetch Wikipedia pages and return chunked LangChain documents with source metadata."""
    pages: list[tuple[str, str]] = []
    for topic in REAL_ESTATE_TOPICS:
        result = _fetch_page(topic)
        if result:
            pages.append(result)
            logger.info("Fetched: %s", result[0])
        else:
            logger.warning("Skipped: %s", topic)

    if not pages:
        raise RuntimeError("No Wikipedia content fetched. Check network connectivity.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

    all_chunks: list[Document] = []
    for title, content in pages:
        text = f"# {title}\n\n{content}"
        docs = splitter.create_documents(
            [text],
            metadatas=[{"source": f"Wikipedia: {title}"}],
        )
        all_chunks.extend(docs)

    logger.info("Produced %d chunks from %d pages.", len(all_chunks), len(pages))
    return all_chunks

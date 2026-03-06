"""
Document ingestion pipeline.

Loads PDFs and Jupyter notebooks from data/, splits them into chunks,
embeds with Ollama nomic-embed-text, and stores in ChromaDB.
Also saves chunks to chunks.pkl for the BM25 hybrid retriever in agent.py.
"""
import logging
import os
import pickle
import re
import shutil

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # suppress ChromaDB posthog noise

from langchain_chroma import Chroma
from langchain_community.document_loaders import NotebookLoader, PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
CHUNKS_PATH = os.path.join(BASE_DIR, "chunks.pkl")
EMBEDDING_MODEL = "nomic-embed-text"


def enrich_metadata(doc, file_path: str):
    """Add source_type, course_week, topic from filename (e.g. C1_W1_Lab01_*.ipynb)."""
    meta = dict(doc.metadata) if doc.metadata else {}
    filename = os.path.basename(file_path)
    meta["source_type"] = "notebook" if filename.endswith(".ipynb") else "pdf"
    match = re.match(r"(C\d+_W\d+)", filename, re.IGNORECASE)
    meta["course_week"] = match.group(1) if match else "unknown"
    if "Lab" in filename:
        lab_match = re.search(r"Lab(\d+)", filename, re.IGNORECASE)
        meta["topic"] = f"Lab{lab_match.group(1)}" if lab_match else "lab"
    elif "slides" in filename.lower():
        meta["topic"] = "slides"
    else:
        meta["topic"] = "other"
    doc.metadata = meta
    return doc


def load_documents():
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    documents = []
    for file in sorted(os.listdir(DATA_DIR)):
        file_path = os.path.join(DATA_DIR, file)
        try:
            if file.endswith(".pdf"):
                logger.info("Loading PDF: %s", file)
                docs = PyPDFLoader(file_path).load()
                for d in docs:
                    enrich_metadata(d, file_path)
                documents.extend(docs)
            elif file.endswith(".ipynb"):
                logger.info("Loading notebook: %s", file)
                docs = NotebookLoader(
                    file_path, include_outputs=False, remove_newline=True
                ).load()
                for d in docs:
                    enrich_metadata(d, file_path)
                documents.extend(docs)
        except Exception:
            logger.exception("Failed to load %s — skipping", file)

    return documents


def split_text(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    return splitter.split_documents(documents)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    raw_docs = load_documents()
    logger.info("Loaded %d document pages/cells.", len(raw_docs))

    chunks = split_text(raw_docs)
    logger.info("Split into %d chunks.", len(chunks))

    emb = OllamaEmbeddings(model=EMBEDDING_MODEL)

    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        logger.info("Removed existing Chroma DB.")

    Chroma.from_documents(chunks, emb, persist_directory=CHROMA_PATH)
    logger.info("Saved %d chunks to %s", len(chunks), CHROMA_PATH)

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    logger.info("Saved chunks to %s for BM25 retriever", CHUNKS_PATH)


if __name__ == "__main__":
    main()

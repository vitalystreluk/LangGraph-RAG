import os
import pickle
import re
from langchain_community.document_loaders import PyPDFLoader, NotebookLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
CHUNKS_PATH = os.path.join(BASE_DIR, "chunks.pkl")  # For BM25 hybrid search
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
    documents = []
    for file in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, file)
        if file.endswith(".pdf"):
            print(f"Loading PDF: {file}")
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            for d in docs:
                enrich_metadata(d, file_path)
            documents.extend(docs)
        elif file.endswith(".ipynb"):
            print(f"Loading Notebook: {file}")
            loader = NotebookLoader(file_path, include_outputs=False, remove_newline=True)
            docs = loader.load()
            for d in docs:
                enrich_metadata(d, file_path)
            documents.extend(docs)
    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    return text_splitter.split_documents(documents)

def main():
    # 1. Load
    raw_docs = load_documents()
    print(f"Loaded {len(raw_docs)} document pages/cells.")

    # 2. Split
    chunks = split_text(raw_docs)
    print(f"Split into {len(chunks)} chunks.")

    # 3. Embed and Store
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    # Clean up existing DB if any
    if os.path.exists(CHROMA_PATH):
        import shutil
        shutil.rmtree(CHROMA_PATH)
        print("Cleaned up existing Chroma DB.")

    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=CHROMA_PATH
    )
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}")

    # Save chunks for BM25 hybrid search (in-memory index at agent startup)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved chunks to {CHUNKS_PATH} for BM25 retriever")

if __name__ == "__main__":
    main()

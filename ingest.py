import os
from langchain_community.document_loaders import PyPDFLoader, NotebookLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# Configuration
BASE_DIR = "/Users/vitalistreliuk/Projects/langgraph-edu/langchain-academy/rag_agent"
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
EMBEDDING_MODEL = "nomic-embed-text"

def load_documents():
    documents = []
    for file in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, file)
        if file.endswith(".pdf"):
            print(f"Loading PDF: {file}")
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        elif file.endswith(".ipynb"):
            print(f"Loading Notebook: {file}")
            # NotebookLoader might need additional care for structured content
            loader = NotebookLoader(file_path, include_outputs=False, remove_newline=True)
            documents.extend(loader.load())
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

if __name__ == "__main__":
    main()

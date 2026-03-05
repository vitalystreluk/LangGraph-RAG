# Agentic ML Research RAG Assistant (Andrew Ng's Specialization)

This project aims to build a sophisticated RAG (Retrieval-Augmented Generation) system that helps users navigate concepts from Andrew Ng's "Supervised Machine Learning" course. Unlike simple RAG, this system will use an agentic approach with self-correction logic, inspired by the "Skeleton Indexing" and "Agentic Router" concepts from the Habr article.

## Proposed Architecture

The system will be built using **LangGraph** with a local **Ollama** backend.

### Components

1.  **Data Source**: PDFs/Slides from Andrew Ng's ML Specialization (Course 1).
2.  **Vector Store**: [ChromaDB](https://www.trychroma.com/) or [FAISS](https://github.com/facebookresearch/faiss) (local).
3.  **Embeddings**: `nomic-embed-text` (running via Ollama).
4.  **LLM**: `qwen2.5:7b` (running via Ollama).
5.  **Logic (The Graph)**:
    *   **Retrieve**: Fetches top-k chunks from the vector store.
    *   **Grade Docs**: A node where the LLM evaluates if the retrieved chunks are relevant to the user's question.
    *   **Generate**: If relevant, generate the answer with citations.
    *   **Transform Query**: If docs are irrelevant, the LLM rewrites the question to improve search results.
    *   **Web Search (Fallback)**: If local docs don't suffice after transformation, search the web (Tavily/DuckDuckGo).

## Implementation Steps

### 1. Environment Setup
*   Install necessary libraries: `chromadb`, `pypdf`, `langchain-community`.
*   Download embedding model: `ollama pull nomic-embed-text`.

### 2. Knowledge Base Indexing
*   Target directory: `rag_agent/data/`.
*   Script to load PDFs, chunk them (RecursiveCharacterTextSplitter), and store them in ChromaDB.

### 3. Graph Definition
*   Define the `State` (messages, documents, relevance flag).
*   Create nodes: `retrieve`, `grade_documents`, `generate`, `transform_query`, `web_search`.
*   Define conditional edges (e.g., `is_relevant -> generate`, `not_relevant -> transform_query` -> `web_search`).

### 4. Verification
*   Test with specific ML questions like "What is Gradient Descent?" or "Difference between L1 and L2 regularization?".
*   Observe the agent's ability to self-correct when the initial retrieval is poor.

## Manual Verification
*   Run the agent via a standalone script or a Jupyter notebook.
*   Check the LangSmith traces (or local logs) to see the decision-making process.

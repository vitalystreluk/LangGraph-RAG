# RAG Agent — Andrew Ng ML Course Assistant

Agentic RAG system built with **LangGraph** and **LangChain** that answers questions about Andrew Ng's "Supervised Machine Learning" course. Demonstrates production-oriented RAG patterns: hybrid search, self-correction, and hallucination detection.

## Tech Stack

| Layer | Technology |
|-------|------------|
| Orchestration | LangGraph (StateGraph, conditional edges) |
| LLM & Embeddings | Ollama (qwen2.5:7b, nomic-embed-text) |
| Vector Store | ChromaDB |
| Retrieval | LangChain retrievers (BM25, Ensemble, MultiQuery) |
| Web Fallback | Tavily API |

## RAG Architecture

The pipeline follows an **Agentic RAG** pattern with built-in self-correction:

```
Retrieve → Grade Documents → [Generate | Transform Query | Web Search] → Grade Generation → Answer
```

- **Hybrid Search**: BM25 + vector similarity (EnsembleRetriever) for keyword and semantic overlap
- **Multi-Query Expansion**: LLM generates query variants to improve recall
- **Per-Document Grading**: Filters irrelevant chunks before generation
- **Hallucination Check**: Verifies the answer is grounded in retrieved documents
- **Query Transformation**: Rewrites the question if initial retrieval is weak
- **Web Search Fallback**: Uses Tavily when local docs are insufficient
- **Conversation Memory**: MemorySaver checkpointing for thread-scoped state

## Project Structure

```
rag_agent/
├── agent.py       # LangGraph workflow, nodes, retriever config
├── ingest.py      # Document loading, chunking, indexing
├── chat_cli.py    # Interactive CLI
├── data/          # PDFs and notebooks (Andrew Ng course)
└── chroma_db/     # Vector store (created by ingest)
```

## Setup

### 1. Dependencies

From the `langchain-academy` root:

```bash
pip install -r rag_agent/requirements.txt
```

Or install in the project venv:

```bash
cd langchain-academy
lc-academy-env/bin/pip install -r rag_agent/requirements.txt
```

### 2. Ollama

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:7b
ollama serve   # if not already running
```

### 3. Optional: Tavily (Web Search Fallback)

Create `rag_agent/.env`:

```
TAVILY_API_KEY=your_key
```

### 4. Index Documents

```bash
cd langchain-academy
python rag_agent/ingest.py
```

## Run

**Interactive chat:**

```bash
cd langchain-academy
python rag_agent/chat_cli.py
```

**Faster mode** (skips grading, MultiQuery, hallucination check — ~3–5× faster):

```bash
RAG_FAST_MODE=1 python rag_agent/chat_cli.py
```

**Single question (programmatic):**

```python
from rag_agent.agent import app, DEFAULT_CONFIG

result = app.invoke({"question": "What is gradient descent?"}, config=DEFAULT_CONFIG)
print(result["generation"])
```

## Data Source

Slides and labs from Andrew Ng's Supervised Machine Learning (Regression and Classification), e.g. [MirandaZhao/Supervised-Machine-Learning-Regression-and-Classification-by-Andrew-Ng](https://github.com/MirandaZhao/Supervised-Machine-Learning-Regression-and-Classification-by-Andrew-Ng). Place PDFs and notebooks in `rag_agent/data/` before running ingest.

---
*Last updated: 2026-03-05*

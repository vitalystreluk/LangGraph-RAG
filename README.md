
Обучающий пилот-проект при прохождении курса [LangChain Academy LangGraph 📸 ](assets/langchain_academy_course.png) и ML от Andrew Ng.
RAG-подходы отчасти взяты из статьи: [Graph RAG — Habr](https://habr.com/ru/articles/1003064/)


-----
# RAG Agent — Andrew Ng ML Course Assistant

Agentic RAG system built with **LangGraph** and **LangChain** that answers questions about Andrew Ng's "Supervised Machine Learning" course. Showcases production-oriented RAG patterns and progressive LangGraph feature usage across two phases.

## Tech Stack

| Layer | Technology |
|-------|------------|
| Orchestration | LangGraph 1.x (StateGraph, subgraphs, Send(), HITL, checkpointing) |
| LLM & Embeddings | Ollama (qwen2.5:7b, nomic-embed-text) |
| Vector Store | ChromaDB |
| Hybrid Retrieval | BM25 + vector (EnsembleRetriever), optional FlashrankRerank |
| Web Fallback | DuckDuckGo (free, no API key) / Tavily fallback |
| Tests | pytest (18 unit tests, no Ollama required) |

## Architecture

```
START
  └─► retrieve  ────────────────────────────────────── retrieval subgraph
  │     expand_queries (LLM generates N variants)
  │       └─► Send() × N ──► retrieve_single (parallel)
  │                                └─► grade_documents
  │
  ├─► generate  ────────────────────────────────────── generation subgraph
  │     generate → grade_generation (hallucination check)
  │
  ├─► transform_query → retrieve  (loop, up to MAX_LOOP_COUNT=2)
  │
  └─► web_search_confirm  ◄── HITL interrupt (user approves)
        └─► web_search → generate
```

## LangGraph Features Demonstrated

### Phase 1 — Core & Code Quality
- **`Annotated[list, operator.add]` state reducers** — nodes return only new step names; LangGraph merges automatically
- **Proper imports** — `langchain.retrievers` (fixes `langchain_classic` bug from v1)
- **Structured logging** — `logging` module instead of `print()`
- **Prompts extracted** to `prompts.py` — separated from graph logic
- **Package structure** — `__init__.py`, no `sys.path` hacks
- **18 pytest unit tests** — run offline, no Ollama needed

### Phase 2 — Advanced LangGraph
- **Subgraphs** (`subgraphs/retrieval.py`, `subgraphs/generation.py`) — compiled sub-StateGraphs as reusable units
- **`Send()` / map-reduce parallel retrieval** — `expand_queries` fans out N parallel `retrieve_single` calls; `raw_documents: Annotated[list, operator.add]` accumulates results; `grade_documents` runs once on the merged pool
- **Human-in-the-loop** — `interrupt_before=["web_search_confirm"]`; CLI asks the user before web search
- **`astream_events(version="v2")`** — token-level streaming with node progress labels
- **`MemorySaver` checkpointing** — thread-scoped conversation memory; state persists between turns
- **`langgraph.json`** — ready for LangGraph Studio

## Project Structure

```
rag_agent/
├── agent.py              # Main graph (composes subgraphs, HITL)
├── state.py              # GraphState, RetrievalState, GenerationState
├── prompts.py            # All LLM prompts
├── tools.py              # Web search tool factory
├── ingest.py             # Document loading, chunking, indexing
├── chat_cli.py           # Interactive CLI (streaming v2, HITL flow)
├── subgraphs/
│   ├── retrieval.py      # Parallel Send() retrieval subgraph
│   └── generation.py     # Generate + hallucination check subgraph
├── tests/
│   └── test_nodes.py     # 18 unit tests (no Ollama required)
├── data/                 # PDFs and notebooks (Andrew Ng course)
├── chroma_db/            # Vector store (created by ingest.py)
├── langgraph.json        # LangGraph Studio config
└── requirements.txt      # Pinned versions
```

## Setup

### 1. Dependencies

```bash
pip install -r requirements.txt
```

### 2. Ollama

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:7b
ollama serve   # if not already running
```

### 3. Web search (optional)

**DuckDuckGo — free, no API key:**
```bash
pip install duckduckgo-search ddgs
```

**Tavily — alternative (requires API key):**
```bash
pip install langchain-tavily
```
Create `.env`:
```
TAVILY_API_KEY=your_key
```

The agent auto-selects: DuckDuckGo → Tavily → disabled. Priority order is defined in `tools.py`.

### 4. Index Documents

```bash
python -m rag_agent.ingest
```

## Run

**Interactive chat (with streaming + HITL):**
```bash
python -m rag_agent.chat_cli
```

**Run tests (no Ollama needed):**
```bash
pytest tests/ -v
```

**Single question (programmatic):**
```python
from rag_agent.agent import app, DEFAULT_CONFIG

result = app.invoke(
    {"question": "What is gradient descent?", "steps": [], "loop_count": 0,
     "documents": [], "generation": "", "relevance": ""},
    config=DEFAULT_CONFIG,
)
print(result["generation"])
print(result["steps"])   # ["retrieve", "generate"]
```

## How HITL Works

When the local knowledge base can't answer the question (after `MAX_LOOP_COUNT=2` query rewrites), the graph pauses at `web_search_confirm` before hitting the internet:

```
🔍 Retrieving documents...
🔄 Rephrasing question...
🔍 Retrieving documents...
⏸  Paused — web search needed

⚠️  The local knowledge base didn't have a good answer.
   The agent wants to search the web.
   Allow web search? [y/n]: y

🌐 Searching the web...
💡 ANSWER:
...
```

Type `n` to skip and get an answer from partial context.

## Data Source

Slides and labs from Andrew Ng's Supervised Machine Learning course. Place PDFs and notebooks in `data/` before running `ingest`.

---
*Last updated: 2026-03-04*

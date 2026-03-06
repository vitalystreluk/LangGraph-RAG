"""
Main RAG agent graph — Phase 1 + Phase 2.

Graph flow:
    START
      └─► retrieve          (calls retrieval subgraph: expand→Send()×N→grade)
            ├─► generate    (if relevant docs found)
            ├─► transform_query → retrieve  (loop, up to MAX_LOOP_COUNT)
            └─► web_search_confirm  ◄── HITL interrupt_before
                    └─► web_search
                            └─► generate
      generate ──► END

Phase 2 features demonstrated:
  - Subgraphs: retrieval and generation are compiled sub-StateGraphs
  - Parallel retrieval: Send() fan-out inside retrieval subgraph
  - Human-in-the-loop: interrupt_before=["web_search_confirm"]
  - Streaming v2: consumed by chat_cli.py via astream_events(version="v2")
  - MemorySaver checkpointing: thread-scoped conversation memory
"""
import logging
import os
import pickle

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# Suppress telemetry noise when offline
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("LANGSMITH_TRACING", "false")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")   # ChromaDB posthog

from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from rag_agent.prompts import TRANSFORM_QUERY_PROMPT
from rag_agent.state import GraphState, doc_to_dict
from rag_agent.subgraphs.generation import build_generation_graph
from rag_agent.subgraphs.retrieval import build_retrieval_graph
from rag_agent.tools import create_web_search_tool, invoke_web_search

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
CHUNKS_PATH = os.path.join(BASE_DIR, "chunks.pkl")
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:7b"
MAX_LOOP_COUNT = 2   # max transform_query retries before falling back to web search

# ── LLM and embeddings ────────────────────────────────────────────────────────
llm = ChatOllama(model=LLM_MODEL, temperature=0)
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# ── Hybrid retriever (BM25 + vector via EnsembleRetriever) ───────────────────
# Fix from Phase 1: correct package is `langchain.retrievers`, not `langchain_classic`
_base_retriever = vector_retriever
try:
    from langchain.retrievers import EnsembleRetriever           # noqa: PLC0415
    from langchain_community.retrievers import BM25Retriever     # noqa: PLC0415

    if os.path.exists(CHUNKS_PATH):
        with open(CHUNKS_PATH, "rb") as _f:
            _chunks = pickle.load(_f)
        _bm25 = BM25Retriever.from_documents(_chunks, k=10)
        _base_retriever = EnsembleRetriever(
            retrievers=[_bm25, vector_retriever],
            weights=[0.4, 0.6],
        )
        logger.info("Hybrid retriever: BM25 + vector (EnsembleRetriever)")
    else:
        logger.info("chunks.pkl not found — using vector-only retriever")
except (ImportError, Exception) as e:
    logger.info("BM25/EnsembleRetriever unavailable (%s) — using vector-only", e)

# ── Optional FlashrankRerank ──────────────────────────────────────────────────
_retriever_after_rerank = _base_retriever
try:
    from langchain.retrievers import ContextualCompressionRetriever  # noqa: PLC0415
    from langchain_community.document_compressors import FlashrankRerank  # noqa: PLC0415

    _reranker = FlashrankRerank(top_n=5)
    _retriever_after_rerank = ContextualCompressionRetriever(
        base_compressor=_reranker,
        base_retriever=_base_retriever,
    )
    logger.info("Reranker: FlashrankRerank enabled")
except Exception as e:
    logger.info("FlashrankRerank unavailable (%s) — skipping", e)

retriever = _retriever_after_rerank

# ── Build compiled subgraphs ──────────────────────────────────────────────────
retrieval_app = build_retrieval_graph(llm, retriever)
generation_app = build_generation_graph(llm)
web_search_tool = create_web_search_tool()

# ── Parent graph nodes ─────────────────────────────────────────────────────────

def retrieve(state: GraphState) -> dict:
    """
    Wrapper that calls the retrieval subgraph.

    The subgraph internally runs:
      expand_queries → [retrieve_single × N via Send()] → grade_documents
    """
    logger.info("[agent] retrieve")
    result = retrieval_app.invoke({
        "question": state["question"],
        "query_variants": [],
        "raw_documents": [],
        "documents": [],
        "relevance": "",
    })
    return {
        "documents": result["documents"],
        "relevance": result["relevance"],
        "steps": ["retrieve"],
    }


def generate(state: GraphState) -> dict:
    """
    Wrapper that calls the generation subgraph.

    The subgraph internally runs: generate → grade_generation (hallucination check).
    """
    logger.info("[agent] generate")
    result = generation_app.invoke({
        "question": state["question"],
        "documents": state["documents"],
        "generation": "",
    })
    return {
        "generation": result["generation"],
        "steps": ["generate"],
    }


def transform_query(state: GraphState) -> dict:
    """Rewrite the question to improve retrieval recall."""
    logger.info("[agent] transform_query")
    chain = TRANSFORM_QUERY_PROMPT | llm
    response = chain.invoke({"question": state["question"]})
    better_q = response.content.strip().split("\n")[0]
    logger.info("[agent] new question: %s", better_q)
    return {
        "question": better_q,
        "loop_count": state.get("loop_count", 0) + 1,
        "steps": ["transform_query"],
    }


def web_search_confirm(state: GraphState) -> dict:
    """
    No-op node that serves as the HITL interrupt point.

    The graph is compiled with interrupt_before=["web_search_confirm"].
    Execution pauses here; chat_cli.py asks the user for confirmation,
    then resumes via app.invoke(None, config=config).
    """
    logger.info("[agent] web_search_confirm (HITL interrupt)")
    return {}


def web_search(state: GraphState) -> dict:
    """Fall back to Tavily web search when local docs are insufficient."""
    logger.info("[agent] web_search")
    question = state["question"]
    content = invoke_web_search(web_search_tool, question)

    if content:
        web_doc = {"page_content": content, "metadata": {"source": "web_search"}}
        new_docs = list(state.get("documents") or []) + [web_doc]
    else:
        logger.warning("[agent] web search returned no results")
        new_docs = list(state.get("documents") or [])

    return {"documents": new_docs, "steps": ["web_search"]}


# ── Conditional edges ──────────────────────────────────────────────────────────

def decide_to_generate(state: GraphState) -> str:
    """
    After retrieval (or transform+retrieval), decide what to do next.

    Routing logic:
      - relevant docs found            → generate
      - no relevant docs, retries left → transform_query (rewrites question, re-retrieves)
      - no relevant docs, retries used → web_search_confirm (HITL pause before web fallback)
    """
    if state.get("relevance") == "yes":
        return "generate"
    if state.get("loop_count", 0) < MAX_LOOP_COUNT:
        return "transform_query"
    return "web_search_confirm"


# ── Build main graph ──────────────────────────────────────────────────────────
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("transform_query", transform_query)
workflow.add_node("web_search_confirm", web_search_confirm)
workflow.add_node("web_search", web_search)
workflow.add_node("generate", generate)

workflow.add_edge(START, "retrieve")
workflow.add_conditional_edges(
    "retrieve",
    decide_to_generate,
    {
        "generate": "generate",
        "transform_query": "transform_query",
        "web_search_confirm": "web_search_confirm",
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_edge("web_search_confirm", "web_search")
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

# Compile with:
#   - MemorySaver: thread-scoped conversation memory (checkpointing)
#   - interrupt_before: HITL pause before web_search_confirm node
memory = MemorySaver()
app = workflow.compile(
    checkpointer=memory,
    interrupt_before=["web_search_confirm"],
)

DEFAULT_CONFIG = {"configurable": {"thread_id": "default"}}

# ── Quick smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = app.invoke(
        {"question": "What is the cost function for linear regression?",
         "steps": [], "loop_count": 0, "documents": [], "generation": "", "relevance": ""},
        config=DEFAULT_CONFIG,
    )
    print("\nFINAL ANSWER:\n", result["generation"])
    print("\nSTEPS:", result["steps"])

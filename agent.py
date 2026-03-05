import os
import pickle
from typing import List, NotRequired, TypedDict

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# Disable LangSmith/PostHog telemetry (avoids timeouts when offline)
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("LANGSMITH_TRACING", "false")

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
try:
    from langchain_tavily import TavilySearch
    web_search_tool = TavilySearch(k=3)
except ImportError:
    from langchain_community.tools.tavily_search import TavilySearchResults
    web_search_tool = TavilySearchResults(k=3)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph
from pydantic import BaseModel, Field

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
CHUNKS_PATH = os.path.join(BASE_DIR, "chunks.pkl")
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:7b"
# Fast mode: skip MultiQuery, grade_documents, grade_generation → 1 embed + 1 LLM instead of 4+4
FAST_MODE = os.environ.get("RAG_FAST_MODE", "").lower() in ("1", "true", "yes")

# --- State Definition ---
MAX_LOOP_COUNT = 2  # Max transform_query iterations before falling back to web_search

# Store documents as list of dicts (msgpack-serializable for MemorySaver checkpointer)
def _to_serializable(val):
    """Convert numpy/other types to msgpack-serializable Python types."""
    try:
        import numpy as np
        if isinstance(val, np.ndarray):
            return val.tolist()
        if hasattr(val, "item"):  # numpy scalar (float32, int64, etc.)
            return val.item()
    except ImportError:
        pass
    if isinstance(val, (list, tuple)):
        return [_to_serializable(v) for v in val]
    if isinstance(val, dict):
        return {k: _to_serializable(v) for k, v in val.items()}
    if isinstance(val, (str, int, float, bool, type(None))):
        return val
    return str(val)


def _doc_to_dict(doc: Document) -> dict:
    meta = dict(doc.metadata) if doc.metadata else {}
    return {"page_content": doc.page_content, "metadata": _to_serializable(meta)}


def _dict_to_doc(d: dict) -> Document:
    return Document(page_content=d["page_content"], metadata=d.get("metadata", {}))


def _get_page_content(item) -> str:
    return item["page_content"] if isinstance(item, dict) else item.page_content


class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[dict]  # [{"page_content": str, "metadata": dict}, ...] — serializable
    steps: List[str]
    loop_count: int
    relevance: NotRequired[str]  # Set by grade_documents: "yes" or "no"

# --- LLM and Tools ---
llm = ChatOllama(model=LLM_MODEL, temperature=0)
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5 if FAST_MODE else 10})

# Hybrid search, re-ranking, multi-query (optional — fallback to vector-only if packages unavailable)
# In fast mode: vector only, no BM25/MultiQuery/Flashrank
_base_retriever = vector_retriever
if not FAST_MODE:
    try:
        from langchain_community.retrievers import BM25Retriever
        from langchain_classic.retrievers import EnsembleRetriever
        if os.path.exists(CHUNKS_PATH):
            with open(CHUNKS_PATH, "rb") as f:
                _chunks = pickle.load(f)
            bm25_retriever = BM25Retriever.from_documents(_chunks, k=10)
            _base_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=[0.4, 0.6],
            )
    except (ImportError, FileNotFoundError):
        pass  # use vector_retriever

_retriever_after_rerank = _base_retriever
if not FAST_MODE:
    try:
        from langchain_community.document_compressors import FlashrankRerank
        from langchain_classic.retrievers import ContextualCompressionRetriever
        reranker = FlashrankRerank(top_n=5)
        _retriever_after_rerank = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=_base_retriever,
        )
    except Exception:
        pass  # Flashrank may fail (model download, ONNX, etc) — use base retriever

retriever = _retriever_after_rerank
if not FAST_MODE:
    try:
        from langchain_classic.retrievers import MultiQueryRetriever
        retriever = MultiQueryRetriever.from_llm(
            retriever=_retriever_after_rerank,
            llm=llm,
            include_original=True,
        )
    except ImportError:
        pass

# --- Pydantic models for structured output ---
class GradeDocumentsResult(BaseModel):
    """Per-document relevance scores."""
    scores: List[str] = Field(
        description="List of 'yes' or 'no' for each document in order. One score per document."
    )


class GradeGeneration(BaseModel):
    """Check if generated answer is grounded in the retrieved documents."""
    is_grounded: str = Field(
        description="'yes' if the answer is supported by the documents, 'no' if it contains unsupported claims"
    )

# --- Nodes ---

def retrieve(state):
    """Retrieve documents from vectorstore."""
    print("---RETRIEVE---")
    question = state["question"]
    docs = retriever.invoke(question)
    documents = [_doc_to_dict(d) for d in docs]
    loop_count = state.get("loop_count", 0)
    return {"documents": documents, "question": question, "steps": state.get("steps", []) + ["retrieve"], "loop_count": loop_count}

def generate(state):
    """Generate answer."""
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    context = "\n\n".join([_get_page_content(d) for d in documents])
    prompt = ChatPromptTemplate.from_template(
        "You are an assistant for question-answering tasks focusing on Andrew Ng's Machine Learning course.\n"
        "Use the following pieces of retrieved context to answer the question.\n"
        "If you don't know the answer, just say that you don't know.\n"
        "Context: {context}\n"
        "Question: {question}\n"
        "Answer:"
    )
    
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    return {"generation": response.content, "steps": state["steps"] + ["generate"]}


def grade_generation(state):
    """Checks if the generated answer is grounded in the retrieved documents."""
    print("---CHECK HALLUCINATION (GROUNDEDNESS)---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    if not documents:
        return {"steps": state["steps"] + ["grade_generation"]}

    context = "\n\n".join([_get_page_content(d) for d in documents])
    prompt = ChatPromptTemplate.from_template(
        "You are a fact-checker. Determine if the following answer is fully supported by the provided context.\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer: {generation}\n\n"
        "Is the answer grounded in the context? Output 'yes' or 'no'."
    )
    structured_llm = llm.with_structured_output(GradeGeneration)
    grader_chain = prompt | structured_llm
    result = grader_chain.invoke({"context": context, "question": question, "generation": generation})

    if result.is_grounded.strip().lower() != "yes":
        disclaimer = "\n\n[Note: This answer may contain information not fully supported by the retrieved documents.]"
        return {"generation": generation + disclaimer, "steps": state["steps"] + ["grade_generation"]}
    return {"steps": state["steps"] + ["grade_generation"]}


def grade_documents(state):
    """Grades each document individually and filters out irrelevant ones."""
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    if not documents:
        return {"documents": [], "steps": state["steps"] + ["grade_documents"], "relevance": "no"}

    # Build context with numbered documents for per-document grading
    numbered_docs = "\n\n".join(
        f"Document {i+1}:\n{_get_page_content(d)[:500]}..." if len(_get_page_content(d)) > 500 else f"Document {i+1}:\n{_get_page_content(d)}"
        for i, d in enumerate(documents)
    )
    prompt = ChatPromptTemplate.from_template(
        "You are a grader assessing relevance of each retrieved document to a user question.\n"
        "Documents:\n\n{context}\n\n"
        "User question: {question}\n\n"
        "For each document (1 to {num_docs}), output 'yes' if it contains information relevant to the question, else 'no'.\n"
        "Return exactly {num_docs} scores in order."
    )
    structured_llm = llm.with_structured_output(GradeDocumentsResult)
    grader_chain = prompt | structured_llm
    result = grader_chain.invoke({
        "context": numbered_docs,
        "question": question,
        "num_docs": len(documents),
    })

    # Filter to keep only relevant documents
    if len(result.scores) != len(documents):
        scores = (result.scores + ["yes"] * len(documents))[:len(documents)]
    else:
        scores = result.scores
    relevant_docs = [d for d, s in zip(documents, scores) if str(s).strip().lower() == "yes"]

    relevance = "yes" if relevant_docs else "no"
    print(f"---SCORE: {relevance} ({len(relevant_docs)}/{len(documents)} relevant)---")
    return {
        "documents": relevant_docs,
        "steps": state["steps"] + ["grade_documents"],
        "relevance": relevance,
    }

def transform_query(state):
    """Transform the query to produce a better question."""
    print("---TRANSFORM QUERY---")
    question = state["question"]
    
    prompt = ChatPromptTemplate.from_template(
        "You are a query transformer that optimizes a user question for a RAG system.\n"
        "Original question: {question}\n\n"
        "Rule: Output ONLY the improved question text. No introduction. No explanation. No conversation.\n"
        "Improved question:"
    )
    
    chain = prompt | llm
    better_question = chain.invoke({"question": question})
    clean_question = better_question.content.strip().split("\n")[0]
    print(f"---NEW QUESTION: {clean_question}---")
    return {"question": clean_question, "steps": state["steps"] + ["transform_query"], "loop_count": state.get("loop_count", 0) + 1}

def web_search(state):
    """Web search based on the transformed question."""
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    # Web search (TavilySearch/TavilySearchResults return list of dicts or Documents)
    docs = web_search_tool.invoke({"query": question})
    parts = []
    for d in docs:
        if isinstance(d, dict):
            parts.append(d.get("content", d.get("page_content", str(d))))
        else:
            parts.append(getattr(d, "page_content", getattr(d, "content", str(d))))
    web_results_dict = {"page_content": "\n".join(parts), "metadata": {"source": "web_search"}}

    # Create new list to avoid mutating state (LangGraph best practice)
    new_documents = list(documents) if documents else []
    new_documents.append(web_results_dict)
    return {"documents": new_documents, "steps": state["steps"] + ["web_search"]}

# --- Edges ---

def decide_to_generate(state):
    """Determines whether to generate an answer, or re-generate a query."""
    if state.get("relevance") == "yes":
        return "generate"
    # Prevent infinite loops: after MAX_LOOP_COUNT transforms, fall back to web search
    loop_count = state.get("loop_count", 0)
    if loop_count >= MAX_LOOP_COUNT:
        return "web_search"
    # If we already tried transforming, go to web search
    if "transform_query" in state.get("steps", []):
        return "web_search"
    return "transform_query"

# --- Build Graph ---
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("grade_generation", grade_generation)
workflow.add_node("transform_query", transform_query)
workflow.add_node("web_search", web_search)


def _route_after_retrieve(state):
    """In fast mode: retrieve -> generate -> END. Else: retrieve -> grade_documents."""
    return "generate" if FAST_MODE else "grade_documents"


workflow.add_edge(START, "retrieve")
workflow.add_conditional_edges("retrieve", _route_after_retrieve, {"grade_documents": "grade_documents", "generate": "generate"})
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "generate": "generate",
        "transform_query": "transform_query",
        "web_search": "web_search",
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_edge("web_search", "generate")


def _route_after_generate(state):
    """In fast mode: skip grade_generation."""
    return "grade_generation" if not FAST_MODE else "__end__"


workflow.add_conditional_edges("generate", _route_after_generate, {"grade_generation": "grade_generation", "__end__": END})
workflow.add_edge("grade_generation", END)

# Compile with checkpointer for conversation memory (state persistence per thread)
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Default config for thread-scoped conversation memory
DEFAULT_CONFIG = {"configurable": {"thread_id": "default"}}

# --- Test ---
if __name__ == "__main__":
    inputs = {"question": "What is the cost function for linear regression?"}
    final_state = app.invoke(inputs, config=DEFAULT_CONFIG)
    print("\nFINAL ANSWER:\n")
    print(final_state["generation"])

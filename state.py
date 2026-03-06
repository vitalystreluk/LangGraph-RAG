"""
State definitions for the RAG agent graphs.

Design notes:
  - GraphState.steps uses operator.add: every node returns only its NEW step name(s),
    LangGraph appends them automatically — no manual list concatenation.
  - RetrievalState.raw_documents uses operator.add so parallel Send() branches
    accumulate retrieved docs before the grade_documents node runs.
  - Subgraph states are intentionally separate from GraphState to avoid
    field-inheritance side-effects across parent/subgraph boundaries.
"""
import operator
from typing import Annotated, TypedDict

from langchain_core.documents import Document


# ── Serialization helpers ─────────────────────────────────────────────────────

def _to_serializable(val):
    """Recursively convert non-serializable types to plain Python types."""
    if isinstance(val, (list, tuple)):
        return [_to_serializable(v) for v in val]
    if isinstance(val, dict):
        return {k: _to_serializable(v) for k, v in val.items()}
    if isinstance(val, (str, int, float, bool, type(None))):
        return val
    return str(val)


def doc_to_dict(doc) -> dict:
    """Convert a LangChain Document (or passthrough dict) to a serializable dict."""
    if isinstance(doc, dict):
        return doc
    meta = dict(doc.metadata) if doc.metadata else {}
    return {"page_content": doc.page_content, "metadata": _to_serializable(meta)}


def dict_to_doc(d: dict) -> Document:
    return Document(page_content=d["page_content"], metadata=d.get("metadata", {}))


def get_page_content(item) -> str:
    return item["page_content"] if isinstance(item, dict) else item.page_content


# ── Parent graph state ────────────────────────────────────────────────────────

class GraphState(TypedDict):
    """
    Main orchestration state.

    `steps` uses operator.add — nodes return only NEW step name(s), e.g.
        return {"steps": ["retrieve"]}
    and LangGraph appends to the existing list automatically.
    """
    question: str
    generation: str
    documents: list[dict]               # last-value: replaced by each subgraph call
    steps: Annotated[list[str], operator.add]
    loop_count: int
    relevance: str                      # "yes" | "no"


# ── Retrieval subgraph states ─────────────────────────────────────────────────

class RetrieveSingleState(TypedDict):
    """Private state passed to each parallel retrieve_single node via Send()."""
    question: str


class RetrievalState(TypedDict):
    """
    Internal state of the retrieval subgraph.

    `raw_documents` uses operator.add so that all parallel retrieve_single
    branches (launched via Send()) accumulate their results into one list
    before grade_documents runs.
    """
    question: str
    query_variants: list[str]
    raw_documents: Annotated[list[dict], operator.add]  # parallel accumulator
    documents: list[dict]                               # filtered/graded result
    relevance: str


# ── Generation subgraph states ────────────────────────────────────────────────

class GenerationState(TypedDict):
    """Internal state of the generation subgraph."""
    question: str
    documents: list[dict]
    generation: str

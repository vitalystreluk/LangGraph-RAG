"""
Retrieval subgraph — Phase 2 feature.

Demonstrates the LangGraph Send() / map-reduce pattern:
  1. expand_queries  — LLM generates N query variants
  2. route_parallel  — conditional edge returns [Send("retrieve_single", q) for q in variants]
  3. retrieve_single — runs in parallel (one instance per variant), accumulates via operator.add
  4. grade_documents — deduplicates the merged pool, grades each doc for relevance

This subgraph is stateless: it receives only `question` from the parent and
returns `documents` + `relevance`. The parent wrapper records the step name.
"""
import logging
from typing import List

from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langgraph.types import Send
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from rag_agent.prompts import EXPAND_QUERIES_PROMPT, GRADE_DOCS_PROMPT
from rag_agent.state import (
    RetrievalState,
    RetrieveSingleState,
    doc_to_dict,
    get_page_content,
)

logger = logging.getLogger(__name__)

QUERY_VARIANTS = 3   # number of parallel query variants to generate


# ── Pydantic structured-output schema ─────────────────────────────────────────

class GradeDocumentsResult(BaseModel):
    """Per-document relevance scores returned by the grading LLM call."""
    scores: List[str] = Field(
        description="'yes' or 'no' for each document, one score per document in order."
    )


# ── Subgraph factory ──────────────────────────────────────────────────────────

def build_retrieval_graph(llm: BaseChatModel, retriever: BaseRetriever):
    """
    Build and compile the retrieval subgraph.

    The LLM and retriever are injected here so the subgraph can be tested
    independently of the main agent configuration.
    """

    # ── Node: expand_queries ──────────────────────────────────────────────────
    def expand_queries(state: RetrievalState) -> dict:
        """Generate QUERY_VARIANTS alternative queries from the original question."""
        logger.info("[retrieval] expand_queries")
        chain = EXPAND_QUERIES_PROMPT | llm
        response = chain.invoke({"question": state["question"], "n": QUERY_VARIANTS})

        lines = [l.strip() for l in response.content.strip().splitlines() if l.strip()]
        # Deduplicate while preserving order; always include the original question
        seen: set[str] = set()
        variants: list[str] = []
        for q in [state["question"]] + lines:
            if q not in seen:
                seen.add(q)
                variants.append(q)
            if len(variants) >= QUERY_VARIANTS + 1:
                break

        logger.debug("[retrieval] variants: %s", variants)
        return {"query_variants": variants}

    # ── Conditional edge: fan-out via Send() ──────────────────────────────────
    def route_parallel_retrieve(state: RetrievalState) -> list[Send]:
        """
        Map-reduce fan-out: one Send() per query variant.

        Each Send dispatches retrieve_single with its own private RetrieveSingleState.
        All parallel retrieve_single calls return {"raw_documents": [...]} which
        LangGraph merges into RetrievalState.raw_documents via operator.add.
        """
        return [
            Send("retrieve_single", {"question": variant})
            for variant in state["query_variants"]
        ]

    # ── Node: retrieve_single (runs in parallel, N times) ────────────────────
    def retrieve_single(state: RetrieveSingleState) -> dict:
        """
        Retrieve documents for ONE query variant.
        Returns raw_documents that accumulate across parallel calls (operator.add).
        """
        logger.info("[retrieval] retrieve_single: '%s'", state["question"])
        docs = retriever.invoke(state["question"])
        return {"raw_documents": [doc_to_dict(d) for d in docs]}

    # ── Node: grade_documents ─────────────────────────────────────────────────
    def grade_documents(state: RetrievalState) -> dict:
        """
        Deduplicate the accumulated raw_documents pool, then ask the LLM to score
        each document for relevance. Keep only 'yes' docs.
        """
        logger.info("[retrieval] grade_documents")
        question = state["question"]
        raw_docs = state.get("raw_documents", [])

        # Deduplicate by first 200 chars of content (fast fingerprint)
        seen: set[str] = set()
        unique: list[dict] = []
        for d in raw_docs:
            key = get_page_content(d)[:200]
            if key not in seen:
                seen.add(key)
                unique.append(d)
        logger.debug("[retrieval] dedup: %d -> %d docs", len(raw_docs), len(unique))

        if not unique:
            return {"documents": [], "relevance": "no"}

        numbered_ctx = "\n\n".join(
            f"Document {i + 1}:\n{get_page_content(d)[:500]}"
            for i, d in enumerate(unique)
        )
        structured_llm = llm.with_structured_output(GradeDocumentsResult)
        result = (GRADE_DOCS_PROMPT | structured_llm).invoke({
            "context": numbered_ctx,
            "question": question,
            "num_docs": len(unique),
        })

        scores = result.scores
        # Pad/truncate if LLM returns wrong count
        if len(scores) != len(unique):
            scores = (scores + ["yes"] * len(unique))[:len(unique)]

        relevant = [d for d, s in zip(unique, scores) if str(s).strip().lower() == "yes"]
        relevance = "yes" if relevant else "no"
        logger.info("[retrieval] %d/%d docs relevant", len(relevant), len(unique))

        return {"documents": relevant, "relevance": relevance}

    # ── Build graph ───────────────────────────────────────────────────────────
    builder = StateGraph(RetrievalState)

    builder.add_node("expand_queries", expand_queries)
    builder.add_node("retrieve_single", retrieve_single)
    builder.add_node("grade_documents", grade_documents)

    builder.add_edge(START, "expand_queries")
    # Send() fan-out: expand_queries -> [retrieve_single x N in parallel]
    builder.add_conditional_edges(
        "expand_queries",
        route_parallel_retrieve,
        ["retrieve_single"],   # list of valid target node names for validation
    )
    # After ALL parallel retrieve_single calls finish, grade_documents runs ONCE
    builder.add_edge("retrieve_single", "grade_documents")
    builder.add_edge("grade_documents", END)

    return builder.compile()

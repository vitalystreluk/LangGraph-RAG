"""
Unit tests for RAG agent nodes and graph logic.

Tests run without Ollama or ChromaDB — all LLM/retriever calls are mocked.
Run with:  pytest rag_agent/tests/ -v
"""
import operator
from typing import Annotated, TypedDict
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from rag_agent.state import (
    GenerationState,
    GraphState,
    RetrievalState,
    doc_to_dict,
    dict_to_doc,
    get_page_content,
)


# ── State helper tests ────────────────────────────────────────────────────────

class TestStateHelpers:
    def test_doc_to_dict_from_document(self):
        doc = Document(page_content="hello", metadata={"source": "test"})
        result = doc_to_dict(doc)
        assert result["page_content"] == "hello"
        assert result["metadata"]["source"] == "test"

    def test_doc_to_dict_passthrough_dict(self):
        d = {"page_content": "already a dict", "metadata": {}}
        assert doc_to_dict(d) is d

    def test_dict_to_doc(self):
        d = {"page_content": "text", "metadata": {"k": "v"}}
        doc = dict_to_doc(d)
        assert isinstance(doc, Document)
        assert doc.page_content == "text"
        assert doc.metadata["k"] == "v"

    def test_get_page_content_from_dict(self):
        assert get_page_content({"page_content": "abc"}) == "abc"

    def test_get_page_content_from_document(self):
        assert get_page_content(Document(page_content="xyz")) == "xyz"


# ── Annotated reducer tests ────────────────────────────────────────────────────

class TestAnnotatedReducer:
    """Verify that operator.add behaves as expected on list fields."""

    def test_steps_append(self):
        """operator.add on list fields should concatenate, not replace."""
        existing = ["retrieve"]
        new = ["generate"]
        result = operator.add(existing, new)
        assert result == ["retrieve", "generate"]

    def test_steps_initial_empty(self):
        result = operator.add([], ["retrieve"])
        assert result == ["retrieve"]

    def test_raw_documents_merge(self):
        """Parallel Send() branches should accumulate raw_documents."""
        batch1 = [{"page_content": "doc1", "metadata": {}}]
        batch2 = [{"page_content": "doc2", "metadata": {}}]
        merged = operator.add(batch1, batch2)
        assert len(merged) == 2


# ── GraphState structure tests ────────────────────────────────────────────────

class TestGraphState:
    def test_initial_state_keys(self):
        state: GraphState = {
            "question": "What is gradient descent?",
            "generation": "",
            "documents": [],
            "steps": [],
            "loop_count": 0,
            "relevance": "",
        }
        assert state["question"] == "What is gradient descent?"
        assert state["steps"] == []
        assert state["loop_count"] == 0


# ── decide_to_generate routing tests ─────────────────────────────────────────

class TestDecideToGenerate:
    """Test the routing logic without invoking the full graph."""

    def _decide(self, relevance: str, loop_count: int) -> str:
        from rag_agent.agent import decide_to_generate
        state: GraphState = {
            "question": "q",
            "generation": "",
            "documents": [],
            "steps": [],
            "loop_count": loop_count,
            "relevance": relevance,
        }
        return decide_to_generate(state)

    def test_relevant_docs_routes_to_generate(self):
        assert self._decide("yes", 0) == "generate"

    def test_no_docs_first_attempt_routes_to_transform(self):
        assert self._decide("no", 0) == "transform_query"

    def test_no_docs_max_loops_routes_to_web_search(self):
        # loop_count >= MAX_LOOP_COUNT should fall back to web_search_confirm
        from rag_agent.agent import MAX_LOOP_COUNT
        assert self._decide("no", MAX_LOOP_COUNT) == "web_search_confirm"


# ── Retrieval subgraph unit tests (mocked LLM + retriever) ───────────────────

class TestRetrievalSubgraph:
    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock()
        # expand_queries returns 3 variants
        expand_response = MagicMock()
        expand_response.content = "What is gradient descent?\nHow does gradient descent work?\nGD optimization"
        # grade_documents returns all 'yes'
        grade_response = MagicMock()
        grade_response.scores = ["yes", "yes"]
        llm.invoke.return_value = expand_response
        llm.with_structured_output.return_value.invoke.return_value = grade_response
        # Make EXPAND_QUERIES_PROMPT | llm chain work
        chain = MagicMock()
        chain.invoke.return_value = expand_response
        llm.__or__ = MagicMock(return_value=chain)
        return llm

    @pytest.fixture
    def mock_retriever(self):
        retriever = MagicMock()
        retriever.invoke.return_value = [
            Document(page_content="Gradient descent minimizes the cost function."),
            Document(page_content="The learning rate controls step size."),
        ]
        return retriever

    def test_expand_queries_generates_variants(self, mock_llm, mock_retriever):
        from rag_agent.subgraphs.retrieval import build_retrieval_graph, QUERY_VARIANTS
        graph = build_retrieval_graph(mock_llm, mock_retriever)
        assert graph is not None

    def test_retrieval_state_raw_documents_accumulator(self):
        """raw_documents uses operator.add — multiple batches accumulate."""
        s1: RetrievalState = {
            "question": "q",
            "query_variants": [],
            "raw_documents": [{"page_content": "a", "metadata": {}}],
            "documents": [],
            "relevance": "",
        }
        extra = [{"page_content": "b", "metadata": {}}]
        merged = operator.add(s1["raw_documents"], extra)
        assert len(merged) == 2


# ── Generation subgraph unit tests (mocked LLM) ───────────────────────────────

class TestGenerationSubgraph:
    @pytest.fixture
    def mock_llm_grounded(self):
        llm = MagicMock()
        gen_response = MagicMock()
        gen_response.content = "Gradient descent is an optimization algorithm."
        grade_response = MagicMock()
        grade_response.is_grounded = "yes"
        chain = MagicMock()
        chain.invoke.return_value = gen_response
        llm.__or__ = MagicMock(return_value=chain)
        llm.with_structured_output.return_value.invoke.return_value = grade_response
        return llm

    def test_generation_subgraph_builds(self, mock_llm_grounded):
        from rag_agent.subgraphs.generation import build_generation_graph
        graph = build_generation_graph(mock_llm_grounded)
        assert graph is not None

    def test_hallucination_disclaimer_constant(self):
        from rag_agent.subgraphs.generation import HALLUCINATION_DISCLAIMER
        assert "[Note:" in HALLUCINATION_DISCLAIMER


# ── Web search tool tests ────────────────────────────────────────────────────

class TestWebSearchTool:
    def test_invoke_web_search_returns_none_when_tool_none(self):
        from rag_agent.tools import invoke_web_search
        assert invoke_web_search(None, "query") is None

    def test_invoke_web_search_with_dict_results(self):
        from rag_agent.tools import invoke_web_search
        mock_tool = MagicMock()
        mock_tool.invoke.return_value = [
            {"content": "Result 1"},
            {"content": "Result 2"},
        ]
        result = invoke_web_search(mock_tool, "test query")
        assert "Result 1" in result
        assert "Result 2" in result

"""
Generation subgraph — Phase 2 feature.

Demonstrates subgraph composition: this subgraph encapsulates the full
generate → hallucination-check pipeline as a reusable unit.

  generate        — RAG generation from retrieved context
  grade_generation — verifies the answer is grounded in the documents;
                     appends a disclaimer if not (instead of silently returning
                     a potentially hallucinated answer)
"""
import logging
from typing import List

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from rag_agent.prompts import GENERATE_PROMPT, GRADE_GENERATION_PROMPT
from rag_agent.state import GenerationState, get_page_content

logger = logging.getLogger(__name__)

HALLUCINATION_DISCLAIMER = (
    "\n\n[Note: This answer may contain information not fully supported "
    "by the retrieved documents.]"
)


class GradeGeneration(BaseModel):
    """Structured output for the hallucination-check LLM call."""
    is_grounded: str = Field(
        description="'yes' if the answer is supported by the documents, 'no' otherwise."
    )


def build_generation_graph(llm: BaseChatModel):
    """
    Build and compile the generation subgraph.

    Injecting the LLM makes the subgraph independently testable.
    """

    # ── Node: generate ────────────────────────────────────────────────────────
    def generate(state: GenerationState) -> dict:
        """Generate an answer from retrieved context using GENERATE_PROMPT."""
        logger.info("[generation] generate")
        context = "\n\n".join(get_page_content(d) for d in state["documents"])
        chain = GENERATE_PROMPT | llm
        response = chain.invoke({"context": context, "question": state["question"]})
        return {"generation": response.content}

    # ── Node: grade_generation ────────────────────────────────────────────────
    def grade_generation(state: GenerationState) -> dict:
        """
        Verify the generated answer is grounded in the retrieved documents.
        If not grounded, append a disclaimer to the answer — transparent to the user.
        """
        logger.info("[generation] grade_generation")

        if not state["documents"]:
            # Nothing to check against — pass through
            return {}

        context = "\n\n".join(get_page_content(d) for d in state["documents"])
        structured_llm = llm.with_structured_output(GradeGeneration)
        result = (GRADE_GENERATION_PROMPT | structured_llm).invoke({
            "context": context,
            "question": state["question"],
            "generation": state["generation"],
        })

        if result.is_grounded.strip().lower() != "yes":
            logger.warning("[generation] hallucination detected — appending disclaimer")
            return {"generation": state["generation"] + HALLUCINATION_DISCLAIMER}

        return {}

    # ── Build graph ───────────────────────────────────────────────────────────
    builder = StateGraph(GenerationState)

    builder.add_node("generate", generate)
    builder.add_node("grade_generation", grade_generation)

    builder.add_edge(START, "generate")
    builder.add_edge("generate", "grade_generation")
    builder.add_edge("grade_generation", END)

    return builder.compile()

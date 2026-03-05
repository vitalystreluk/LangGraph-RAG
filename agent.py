import os
from typing import List, TypedDict
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langgraph.graph import START, END, StateGraph
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# Configuration
BASE_DIR = "/Users/vitalistreliuk/Projects/langgraph-edu/langchain-academy/rag_agent"
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:7b"

# --- State Definition ---
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]
    steps: List[str]
    loop_count: int  # Added to prevent infinite loops

# --- LLM and Tools ---
llm = ChatOllama(model=LLM_MODEL, temperature=0)
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Web Search Tool
web_search_tool = TavilySearchResults(k=3)

# --- Pydantic models for structured output ---
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

# --- Nodes ---

def retrieve(state):
    """Retrieve documents from vectorstore."""
    print("---RETRIEVE---")
    question = state["question"]
    documents = retriever.invoke(question)
    loop_count = state.get("loop_count", 0)
    return {"documents": documents, "question": question, "steps": state.get("steps", []) + ["retrieve"], "loop_count": loop_count}

def generate(state):
    """Generate answer."""
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    context = "\n\n".join([doc.page_content for doc in documents])
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

def grade_documents(state):
    """Determines whether the retrieved documents are relevant to the question."""
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    
    # Simple heuristic or structured LLM call
    # For speed and local reliability, let's use a clear prompt
    prompt = ChatPromptTemplate.from_template(
        "You are a grader assessing relevance of a retrieved document to a user question.\n"
        "Retrieved documents: \n\n{context}\n\n"
        "User question: {question}\n"
        "If the documents contain keywords or semantic meaning related to the user question, grade them as relevant.\n"
        "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
    )
    
    # We use a structured LLM call here
    structured_llm = llm.with_structured_output(GradeDocuments)
    grader_chain = prompt | structured_llm
    
    context = "\n\n".join([doc.page_content for doc in documents])
    score = grader_chain.invoke({"context": context, "question": question})
    
    print(f"---SCORE: {score.binary_score}---")
    return {"steps": state["steps"] + ["grade_documents"], "relevance": score.binary_score}

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

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results, metadata={"source": "web_search"})
    
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
        
    return {"documents": documents, "steps": state["steps"] + ["web_search"]}

# --- Edges ---

def decide_to_generate(state):
    """Determines whether to generate an answer, or re-generate a query."""
    if state.get("relevance") == "yes":
        return "generate"
    else:
        # If it's the first fail, try to transform and retrieve again
        # If we already tried transforming, go to web search
        if "transform_query" in state["steps"]:
            return "web_search"
        return "transform_query"

# --- Build Graph ---
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)
workflow.add_node("web_search", web_search)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
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
workflow.add_edge("generate", END)

# Compile
app = workflow.compile()

# --- Test ---
if __name__ == "__main__":
    inputs = {"question": "What is the cost function for linear regression?"}
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Node '{key}':")
    
    final_state = app.invoke(inputs)
    print("\nFINAL ANSWER:\n")
    print(final_state["generation"])

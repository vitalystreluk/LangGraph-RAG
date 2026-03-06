"""
All LLM prompts in one place — easy to iterate, test and compare.
"""
from langchain_core.prompts import ChatPromptTemplate

GENERATE_PROMPT = ChatPromptTemplate.from_template(
    "You are an assistant for question-answering tasks about Andrew Ng's Machine Learning course.\n"
    "Use the following retrieved context to answer the question.\n"
    "If the context does not contain the answer, say that you don't know.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

GRADE_DOCS_PROMPT = ChatPromptTemplate.from_template(
    "You are a grader assessing relevance of each retrieved document to a user question.\n\n"
    "Documents:\n{context}\n\n"
    "User question: {question}\n\n"
    "For each document (1 to {num_docs}), output 'yes' if it contains information relevant "
    "to the question, else 'no'.\n"
    "Return exactly {num_docs} scores in order."
)

GRADE_GENERATION_PROMPT = ChatPromptTemplate.from_template(
    "You are a fact-checker. Determine if the following answer is fully supported by the context.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer: {generation}\n\n"
    "Is the answer grounded in the context? Output 'yes' or 'no'."
)

TRANSFORM_QUERY_PROMPT = ChatPromptTemplate.from_template(
    "You are a query optimizer for a RAG system searching Andrew Ng's ML course materials.\n"
    "Rewrite the question to improve retrieval recall. Be concise and specific.\n\n"
    "Original question: {question}\n\n"
    "Rule: Output ONLY the improved question. No explanation.\n"
    "Improved question:"
)

# Used by the retrieval subgraph to fan out parallel queries via Send()
EXPAND_QUERIES_PROMPT = ChatPromptTemplate.from_template(
    "Generate {n} different search queries to retrieve relevant information for the question below.\n"
    "Each query should focus on a different aspect or phrasing. Output one query per line, no numbering.\n\n"
    "Question: {question}\n\n"
    "Queries:"
)

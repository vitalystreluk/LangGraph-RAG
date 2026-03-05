# Walkthrough: Agentic ML Research RAG Assistant

I have built a self-correcting RAG agent that uses LangGraph to navigate and explain concepts from Andrew Ng's "Supervised Machine Learning" course.

## 1. Data Acquisition & Indexing
I used a `git clone` strategy to fetch PDFs and Jupyter Notebooks from the course repository.
- **Source**: `MirandaZhao/Supervised-Machine-Learning-Regression-and-Classification-by-Andrew-Ng`
- **Total Files**: 29 (Slides + Labs)
- **Indexing**: 866 chunks stored in a local ChromaDB.

## 2. Agent Architecture (The Graph)
The agent follows an "Agentic RAG" pattern:
1.  **Retrieve**: Fetches context from ChromaDB using `nomic-embed-text`.
2.  **Grade**: Checks if documents are actually relevant using `qwen2.5:7b`.
3.  **Generate**: Creates the final answer if relevance is high.
4.  **Web Search (Fallback)**: If local docs don't suffice, it uses **Tavily API** to fetch real-time data from the internet.

## 3. Verification Result (Local vs Web)
### Case 1: Local Knowledge (Andrew Ng Course)
- **Question**: *"What is the cost function for linear regression?"*
- **Result**: `RETRIEVE` -> `SCORE: yes` -> `GENERATE`. (Answered from PDF)

### Case 2: Web Fallback (Real-time info)
- **Question**: *"What is the latest version of LangGraph in 2026?"*
- **Result**: `RETRIEVE` -> `SCORE: no` -> `TRANSFORM` -> `WEB SEARCH` -> `GENERATE`. (Answered from Internet)

## How to Run
You can run the agent anytime using the following command:
```bash
lc-academy-env/bin/python rag_agent/agent.py
```

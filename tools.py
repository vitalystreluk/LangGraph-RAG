"""
Tool setup for the RAG agent.

Web search is optional — the agent degrades gracefully if Tavily is not installed
or TAVILY_API_KEY is not set.
"""
import logging
import warnings
from typing import Optional

logger = logging.getLogger(__name__)


def create_web_search_tool():
    """
    Return a web-search tool, or None if nothing is available.

    Priority:
      1. DuckDuckGo  — free, no API key required (pip install duckduckgo-search)
      2. TavilySearch (langchain-tavily) — requires TAVILY_API_KEY
      3. TavilySearchResults (legacy langchain-community fallback)
    """
    try:
        from langchain_community.tools import DuckDuckGoSearchRun
        logger.info("Web search: using DuckDuckGoSearchRun (no API key needed)")
        return DuckDuckGoSearchRun()
    except (ImportError, Exception) as e:
        logger.debug("DuckDuckGo unavailable: %s", e)

    try:
        from langchain_tavily import TavilySearch
        logger.info("Web search: using langchain_tavily.TavilySearch")
        return TavilySearch(k=3)
    except ImportError:
        pass

    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
        logger.info("Web search: using langchain_community.TavilySearchResults (legacy)")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return TavilySearchResults(k=3)
    except ImportError:
        pass

    logger.warning("No web search tool available — web fallback disabled.")
    return None


def invoke_web_search(tool, query: str) -> Optional[str]:
    """
    Invoke a web-search tool and return concatenated content as a single string.
    Returns None if tool is None.
    """
    if tool is None:
        return None

    raw = tool.invoke({"query": query})

    # Some tool versions return a plain string (e.g. legacy TavilySearchResults)
    if isinstance(raw, str):
        return raw if raw.strip() else None

    # Detect API-level errors returned as {"error": ...}
    if isinstance(raw, dict) and "error" in raw and "results" not in raw:
        logger.warning("Tavily API error: %s", raw["error"])
        return None

    results = raw if isinstance(raw, list) else raw.get("results", [])

    parts: list[str] = []
    for item in results:
        if isinstance(item, dict):
            parts.append(item.get("content", item.get("page_content", str(item))))
        else:
            parts.append(getattr(item, "page_content", getattr(item, "content", str(item))))

    return "\n\n".join(parts) if parts else None

"""
Interactive CLI for the RAG agent.

Phase 2 features:
  - astream_events(version="v2") for token-level streaming
  - Human-in-the-loop: detects when the graph is interrupted before web_search_confirm,
    asks the user for permission, then resumes or skips the web search
"""
import asyncio
import logging

from rag_agent.agent import DEFAULT_CONFIG, app

logger = logging.getLogger(__name__)

EMPTY_INITIAL_STATE = {
    "question": "",
    "generation": "",
    "documents": [],
    "steps": [],
    "loop_count": 0,
    "relevance": "",
}


# ── Streaming helper ──────────────────────────────────────────────────────────

async def stream_once(inputs, config: dict, suppress_nodes: set[str] | None = None) -> str:
    """
    Stream one invocation of the graph, printing node progress + tokens.
    `suppress_nodes` — set of node names whose progress label should be hidden
    (used when resuming after HITL to avoid printing stale labels).
    Returns the last generation text seen, or "" if interrupted.
    """
    generation = ""
    last_node = None
    suppress_nodes = suppress_nodes or set()

    async for event in app.astream_events(inputs, config=config, version="v2"):
        kind = event.get("event")

        # Node start — print a progress label
        if kind == "on_chain_start":
            node = event.get("metadata", {}).get("langgraph_node")
            if node and node != last_node:
                last_node = node
                label = {
                    "retrieve":           "🔍 Retrieving documents...",
                    "transform_query":    "🔄 Rephrasing question...",
                    "web_search_confirm": "⏸  Paused — web search needed",
                    "web_search":         "🌐 Searching the web...",
                    "generate":           "\n💡 ANSWER:",
                }.get(node)
                if label and node not in suppress_nodes:
                    print(label)

        # Token stream — only from the answer-generation node, not graders/query-expanders
        elif kind == "on_chat_model_stream":
            node = event.get("metadata", {}).get("langgraph_node")
            if node != "generate":
                continue
            chunk = event.get("data", {}).get("chunk")
            if chunk is None:
                continue
            content = chunk.content if hasattr(chunk, "content") else chunk.get("content", "")
            if content:
                print(content, end="", flush=True)
                generation += content

    return generation


# ── HITL helper ───────────────────────────────────────────────────────────────

def ask_web_search_permission() -> bool:
    """Ask the user whether to allow the web search fallback."""
    print("\n" + "─" * 40)
    print("⚠️  The local knowledge base didn't have a good answer.")
    print("   The agent wants to search the web (Tavily).")
    while True:
        choice = input("   Allow web search? [y/n]: ").strip().lower()
        if choice in ("y", "yes"):
            return True
        if choice in ("n", "no"):
            return False
        print("   Please enter y or n.")


# ── Main REPL ─────────────────────────────────────────────────────────────────

async def chat(user_input: str, config: dict) -> str:
    """
    Run one full question-answer cycle, handling HITL interrupts.

    LangGraph interrupt flow:
      1. stream_once() streams until the graph hits interrupt_before=["web_search_confirm"]
      2. app.get_state() reveals pending next nodes
      3. We ask the user; they can approve or skip the web search
      4. Resume with app.invoke(None, config) or update state to bypass web search
    """
    inputs = {**EMPTY_INITIAL_STATE, "question": user_input}
    generation = await stream_once(inputs, config)

    # Check whether the graph is waiting at the HITL interrupt
    graph_state = app.get_state(config)
    while graph_state.next:
        pending = list(graph_state.next)
        if "web_search_confirm" in pending:
            if ask_web_search_permission():
                # Resume: suppress the web_search_confirm label (already shown before interrupt)
                generation = await stream_once(None, config, suppress_nodes={"web_search_confirm"})
            else:
                # Skip web search: force relevance="yes" so decide_to_generate routes to generate
                app.update_state(config, {"relevance": "yes", "steps": ["web_search_skipped"]})
                generation = await stream_once(None, config, suppress_nodes={"web_search_confirm"})
        else:
            # Unexpected interrupt — just resume
            generation = await stream_once(None, config)

        graph_state = app.get_state(config)

    # Fallback if streaming produced no tokens (e.g. Ollama didn't stream)
    if not generation:
        final = app.get_state(config).values
        generation = final.get("generation", "")
        if generation:
            print("\n💡 ANSWER:")
            print(generation)

    return generation


def main():
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    print("\n" + "=" * 52)
    print("🎓  ML Research Assistant")
    print("    Knowledge base: Andrew Ng's Supervised ML course")
    print("    Type 'exit' or 'quit' to stop.")
    print("=" * 52 + "\n")

    while True:
        try:
            user_input = input("🤔 Your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye! Happy learning! 👋")
            break

        if not user_input:
            continue

        print()
        try:
            asyncio.run(chat(user_input, DEFAULT_CONFIG))
        except Exception as exc:
            logger.exception("Error during chat: %s", exc)
            print(f"\n⚠️  Error: {exc}")

        print("\n" + "─" * 30 + "\n")


if __name__ == "__main__":
    main()

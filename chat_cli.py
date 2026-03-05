import asyncio
import sys
import os

# Add project root to path so we can import agent from anywhere
RAG_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(RAG_AGENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from rag_agent.agent import app, DEFAULT_CONFIG, FAST_MODE


async def run_with_token_streaming(inputs: dict) -> str:
    """Run the agent with token-level streaming via astream_events."""
    final_generation = ""
    last_node = None

    async for event in app.astream_events(inputs, config=DEFAULT_CONFIG, version="v1"):
        kind = event.get("event")
        if kind == "on_chain_start":
            meta = event.get("metadata", {})
            node = meta.get("langgraph_node")
            if node and node != last_node:
                last_node = node
                if node == "transform_query":
                    print("🔄 Rephrasing your question...")
                elif node == "grade_documents":
                    print("📋 Checking document relevance...")
                elif node == "web_search":
                    print("🌐 Searching the web...")
                elif node == "generate":
                    print("\n💡 ANSWER:")
        elif kind == "on_chat_model_stream":
            data = event.get("data", {})
            chunk = data.get("chunk", {})
            if hasattr(chunk, "content") and chunk.content:
                print(chunk.content, end="", flush=True)
                final_generation += chunk.content
            elif isinstance(chunk, dict) and chunk.get("content"):
                c = chunk["content"]
                print(c, end="", flush=True)
                final_generation += c

    return final_generation


def main():
    print("\n" + "=" * 50)
    print("🎓 Welcome to your ML Research Assistant!")
    print("Knowledge base: Andrew Ng's Supervised Machine Learning")
    if FAST_MODE:
        print("⚡ Fast mode (RAG_FAST_MODE=1): fewer checks, faster responses")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 50 + "\n")

    while True:
        user_input = input("🤔 Your question: ")

        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye! Happy learning! 👋")
            break

        if not user_input.strip():
            continue

        print("\n🔍 Thinking...")

        inputs = {"question": user_input}

        try:
            final_generation = asyncio.run(run_with_token_streaming(inputs))
            if not final_generation:
                # Fallback if no token stream (e.g. Ollama streaming not supported)
                final_state = app.invoke(inputs, config=DEFAULT_CONFIG)
                final_generation = final_state.get("generation", "")
                print(final_generation)
        except Exception as e:
            print(f"\n⚠️ Streaming failed ({e}), falling back to sync...")
            final_state = app.invoke(inputs, config=DEFAULT_CONFIG)
            final_generation = final_state.get("generation", "")
            if not final_generation:
                print("(No response)")
            else:
                print("\n💡 ANSWER:")
                print(final_generation)

        print("\n" + "-" * 30 + "\n")


if __name__ == "__main__":
    main()

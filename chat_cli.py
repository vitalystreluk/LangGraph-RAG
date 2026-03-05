import sys
import os

# Add project root to path so we can import agent from anywhere
PROJECT_ROOT = "/Users/vitalistreliuk/Projects/langgraph-edu/langchain-academy"
sys.path.append(PROJECT_ROOT)

from rag_agent.agent import app

def main():
    print("\n" + "="*50)
    print("🎓 Welcome to your ML Research Assistant!")
    print("Knowledge base: Andrew Ng's Supervised Machine Learning")
    print("Type 'exit' or 'quit' to stop.")
    print("="*50 + "\n")

    while True:
        user_input = input("🤔 Your question: ")
        
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye! Happy learning! 👋")
            break
            
        if not user_input.strip():
            continue

        print("\n🔍 Thinking...")
        
        inputs = {"question": user_input}
        
        # Stream the nodes execution for visual feedback
        final_generation = ""
        for output in app.stream(inputs):
            for key, value in output.items():
                if key == "generate":
                    final_generation = value["generation"]
                elif key == "transform_query":
                    print(f"🔄 I'm rephrasing your question to look deeper: \"{value['question']}\"")
                elif key == "grade_documents":
                    if value.get("relevance") == "no":
                        print("📉 Initial search didn't yield enough info. Trying an optimized search...")
                elif key == "web_search":
                    print("🌐 Still nothing locally. Expanding search to the web...")
        
        print("\n" + "-"*30)
        print("💡 ANSWER:")
        print(final_generation)
        print("-"*30 + "\n")

if __name__ == "__main__":
    main()

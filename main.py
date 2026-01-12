import os
import sys
from dotenv import load_dotenv

# Add the project root to sys.path to allow absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multiagent.graph import app
from langchain_core.messages import HumanMessage

load_dotenv()

def run_research(query: str):
    """Run the multi-agent research graph with a user query."""
    print(f"\n🚀 Starting research for: '{query}'\n")
    
    # Initial state
    inputs = {
        "messages": [HumanMessage(content=query)],
        "selected_documents": []
    }
    
    # Run the graph
    # thread_id is used for persistent state (memory) in Langgraph
    config = {"configurable": {"thread_id": "researcher_1"}}
    
    try:
        for output in app.stream(inputs, config=config):
            for key, value in output.items():
                print(f"\n--- [Node: {key}] ---")
                if "messages" in value:
                    for msg in value["messages"]:
                        # Print the latest message from the agent or supervisor
                        role = "Assistant" if hasattr(msg, 'tool_calls') and not msg.tool_calls else "System/Agent"
                        if isinstance(msg, HumanMessage):
                            role = "Human"
                        
                        content = msg.content
                        if isinstance(content, list):
                            # Extract text from structured content (list of dicts)
                            content = "".join([c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"])
                        
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            content = f"Calling tools: {[tc['name'] for tc in msg.tool_calls]}"
                        
                        if content.strip(): # Only print if there's actual content
                            print(f"[{role}]: {content}")
    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
    else:
        user_query = input("Enter your research query (or 'quit' to exit): ")
    
    if user_query.lower() not in ['quit', 'exit']:
        run_research(user_query)

import operator
from typing import Annotated, Sequence, TypedDict, Union, List
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from multiagent.agents import llm_with_tools, tools

# Define the state for the graph
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    selected_documents: List[str]

# Define the nodes
def supervisor(state: AgentState):
    """
    The supervisor node decides which tool to call or if it's done.
    It is aware of the documents available in the Pinecone index.
    """
    # List of documents pre-indexed in Pinecone (Part 1)
    available_docs = ["Google.pdf"]
    
    system_prompt = (
        "You are a Research Supervisor. You have access to:\n"
        "1. Arxiv search for scientific papers.\n"
        "2. Web search (Tavily) for general information.\n"
        "3. RAG search (Pinecone) for internal documents.\n\n"
        f"Available documents in the internal index: {available_docs}\n\n"
        "DIRECTIONS:\n"
        "- If the user asks for a summary or search of internal data, use the `rag_search` tool.\n"
        "- When you receive tool results (Arxiv, Web, or RAG), DO NOT just say you found information.\n"
        "- INSTEAD, use the retrieved content to provide a detailed, comprehensive answer or summary to the user.\n"
        "- If the retrieved information is insufficient, say so, but always try to synthesize what you have."
    )
    
    messages = [HumanMessage(content=system_prompt)] + list(state["messages"])
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# Define the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("supervisor", supervisor)

# Define tool execution node
tool_node = ToolNode(tools)
workflow.add_node("tools", tool_node)

# Define edges
workflow.add_edge(START, "supervisor")

# Routing logic based on tool calls
def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

workflow.add_conditional_edges(
    "supervisor",
    should_continue,
    {
        "tools": "tools",
        END: END,
    },
)

workflow.add_edge("tools", "supervisor")

# Compile the graph
app = workflow.compile()

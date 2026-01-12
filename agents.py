import os
from typing import Annotated, TypedDict, List
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from pinecone import Pinecone
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0
)

# --- Tools ---

@tool
def arxiv_search(query: str):
    """Search for relevant research papers on Arxiv."""
    arxiv_wrapper = ArxivAPIWrapper(top_k_results=3, doc_content_chars_max=2000)
    arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)
    return arxiv.run(query)

@tool
def web_search(query: str):
    """Conduct online research for broader context using Tavily."""
    tavily = TavilySearchResults(k=3)
    return tavily.run(query)

# Initialize Embedding Model (Must match ingest.py)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

@tool
def rag_search(query: str, filenames: List[str] = None):
    """
    Search your pre-indexed internal documents in Pinecone.
    This tool retrieves actual content from the documents indexed in Part 1.
    """
    index_name = os.getenv("PINECONE_INDEX_NAME", "research-agent-index")
    
    # Initialize VectorStore
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        pinecone_api_key=os.getenv("PINECONE_API_KEY")
    )
    
    # Define metadata filter if filenames are provided
    search_kwargs = {}
    if filenames:
        search_kwargs["filter"] = {"source": {"$in": filenames}} # ingest.py often uses 'source' for filename
    
    # Perform similarity search
    results = vectorstore.similarity_search(query, k=4, **search_kwargs)
    
    if not results:
        return "No relevant information found in the internal documents."
    
    # Format results for the LLM
    context = "\n\n".join([f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}" for doc in results])
    return context

# List of tools for the agents
tools = [arxiv_search, web_search, rag_search]
llm_with_tools = llm.bind_tools(tools)

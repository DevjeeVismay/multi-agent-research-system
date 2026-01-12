import os
import time
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()
# --- Configuration ---
# Replace with your actual keys or set them as environment variables
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

INDEX_NAME = "research-agent-index"
# While your LLM is gemini-2.5-flash, we use a specialized model for embeddings
EMBEDDING_MODEL = "models/text-embedding-004" 

def setup_vector_db(pdf_paths):
    """
    Creates Pinecone index (if needed), processes PDFs, and upserts vectors.
    """
    # 1. Initialize Pinecone Client
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

    # 2. Create Index if it doesn't exist
    # We use cosine metric which is standard for Google's embeddings
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    
    if INDEX_NAME not in existing_indexes:
        print(f"Creating index: {INDEX_NAME}...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=768, # text-embedding-004 uses 768 dimensions
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws", 
                region="us-east-1" # Free tier supports us-east-1
            )
        )
        # Wait for index to be ready
        while not pc.describe_index(INDEX_NAME).status['ready']:
            time.sleep(1)
        print("Index created successfully!")
    else:
        print(f"Index {INDEX_NAME} already exists.")

    # 3. Initialize Embedding Model
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    # 4. Load and Process PDFs
    all_splits = []
    
    for path in pdf_paths:
        print(f"Loading {path}...")
        try:
            loader = PyPDFLoader(path)
            docs = loader.load()
            
            # Split text into chunks for better retrieval
            # Chunk size 1000 is a good balance for RAG
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(docs)
            all_splits.extend(splits)
            print(f"Processed {len(splits)} chunks from {path}")
            
        except Exception as e:
            print(f"Error loading {path}: {e}")

    # 5. Upsert to Pinecone
    if all_splits:
        print(f"Upserting {len(all_splits)} total chunks to Pinecone...")
        vectorstore = PineconeVectorStore.from_documents(
            documents=all_splits,
            embedding=embeddings,
            index_name=INDEX_NAME
        )
        print("Upload complete!")
    else:
        print("No documents were processed.")

if __name__ == "__main__":
    # Add your local PDF paths here
    test_pdfs = [
        "Google.pdf",
        # "document2.pdf",
        # "document3.pdf"
    ]
    
    # Ensure these files exist before running
    valid_pdfs = [p for p in test_pdfs if os.path.exists(p)]
    
    if valid_pdfs:
        setup_vector_db(valid_pdfs)
    else:
        print("Please add actual PDF files to the test_pdfs list.")

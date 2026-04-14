import argparse
import os
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Import Phase 1 and 2 tools
from chunk_and_embed import chunk_text
from ingest_pdf import extract_text_from_pdf

PERSIST_DIRECTORY = "./chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def get_embedding_model():
    """Initializes and returns the local HuggingFace embedding model."""
    print(f"[INFO] Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def store_in_chroma(pdf_path: str):
    """
    Extracts text, creates chunks, and stores them in ChromaDB with metadata.
    """
    print(f"\n--- Storing Document: {pdf_path} ---")
    text = extract_text_from_pdf(pdf_path)
    if not text:
        print("[ERROR] No text extracted. Aborting.")
        return None
    
    chunks = chunk_text(text, chunk_size=1000, chunk_overlap=200)
    print(f"[SUCCESS] Generated {len(chunks)} chunks.")
    
    # Initialize the embedding model
    embeddings = get_embedding_model()
    
    print(f"[INFO] Initializing ChromaDB at '{PERSIST_DIRECTORY}'...")
    # Add metadata to each chunk
    metadatas = [{"source": pdf_path, "chunk_id": i} for i in range(len(chunks))]
    
    # Create the vector store. This processes embeddings and writes them to the local disk.
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
        metadatas=metadatas
    )
    
    print(f"[SUCCESS] Data stored! Chroma database is ready in {PERSIST_DIRECTORY}.")
    return vectorstore

def query_database(query: str, k: int = 3):
    """
    Queries the existing ChromaDB instance for the top k most relevant chunks.
    """
    print(f"\n--- Querying Vector DB ---")
    print(f"Question: '{query}'")
    
    if not os.path.exists(PERSIST_DIRECTORY):
        print(f"[ERROR] Database not found at '{PERSIST_DIRECTORY}'. Please ingest a document first.")
        return
        
    embeddings = get_embedding_model()
    
    # Load the existing database from disk
    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )
    
    print(f"[INFO] Searching for top {k} results...")
    # Perform similarity search
    results = vectorstore.similarity_search(query, k=k)
    
    print("\n[SUCCESS] --- Top Results ---")
    for idx, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "Unknown")
        chunk_id = doc.metadata.get("chunk_id", "Unknown")
        print(f"\nResult #{idx} (Source: {source}, Chunk: {chunk_id}):")
        # Removing excess newlines for terminal readability
        clean_content = doc.page_content.replace("\n", " ").strip()
        print("'" + clean_content + "'")
        print("-" * 50)
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChromaDB Vector Store Integration")
    parser.add_argument("--ingest", type=str, help="Path to PDF to ingest into the database")
    parser.add_argument("--query", type=str, help="Question to query the vector database")
    parser.add_argument("--k", type=int, default=3, help="Number of results to return (default 3)")
    
    args = parser.parse_args()
    
    if args.ingest:
        store_in_chroma(args.ingest)
        
    elif args.query:
        query_database(args.query, k=args.k)
        
    else:
        parser.print_help()

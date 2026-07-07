import argparse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# Import the text extraction function from our previous script
from ingest_pdf import extract_text_from_pdf

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    """
    Splits the cleaned text into manageable chunks using RecursiveCharacterTextSplitter.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks = text_splitter.split_text(text)
    return chunks

def embed_chunks(chunks: list[str], model_name: str = "all-MiniLM-L6-v2"):
    """
    Takes a list of string chunks and converts them into vector embeddings 
    using a locally hosted HuggingFace model.
    """
    # Load the requested HuggingFace model via SentenceTransformers
    print(f"\n[INFO] Loading local embedding model: '{model_name}'...")
    hf_embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    print(f"[INFO] Generating embeddings for {len(chunks)} chunks. This may take a moment...")
    # Generate embeddings layer for all chunks
    embeddings = hf_embeddings.embed_documents(chunks)
    
    return embeddings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunking and Embedding Script")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file for ingestion")
    args = parser.parse_args()
    
    print(f"--- Starting Phase 2: Chunk & Embed ---")
    
    # 1. Extract text using Phase 1 logic
    print("\n=> Step 1: Extracting Code")
    text = extract_text_from_pdf(args.pdf_path)
    
    if not text:
        print("[ERROR] Failed to extract text. Please ensure the PDF is valid.")
        exit(1)
        
    print(f"[SUCCESS] Extracted {len(text)} characters of text.")
    
    # 2. Chunk the text
    print("\n=> Step 2: Chunking Text")
    chunks = chunk_text(text, chunk_size=1000, chunk_overlap=200)
    print(f"[SUCCESS] Document split into {len(chunks)} overlapping chunks.")
    
    if chunks:
        print("\n--- Preview of Chunk #1 ---")
        print(f"'{chunks[0][:200]}...'")
    
    # 3. Generate Embeddings
    print("\n=> Step 3: Generating Vector Embeddings")
    embeddings = embed_chunks(chunks, model_name="all-MiniLM-L6-v2")
    
    if embeddings:
        print(f"\n[SUCCESS] Successfully generated {len(embeddings)} embedding vectors.")
        print(f"[SUCCESS] Each vector has {len(embeddings[0])} dimensions.")
        print("Ready for Vector DB insertion!")

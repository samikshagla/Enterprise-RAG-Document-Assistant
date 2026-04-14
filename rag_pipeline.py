import argparse
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from vector_db import query_database

def generate_answer(query: str, groq_api_key: str):
    """
    Orchestrates the full RAG pipeline:
    1. Retrieves relevant chunks via Phase 3 retrieval
    2. Builds a strict grounding prompt with the context
    3. Queries the Groq LLM for the final answer
    """
    # 1. Retrieve the context
    results = query_database(query, k=5)
    if not results:
        print("[ERROR] No context retrieved. Cannot answer.")
        return
        
    # Extract text from the results
    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
    
    # 2. Initialize the LLM (Groq)
    # Using LLaMA 3 8B as it is fast and excellent for inference
    print("\n[INFO] Connecting to Groq via LLM...")
    llm = ChatGroq(
        api_key=groq_api_key,
        model="llama-3.3-70b-versatile",
        temperature=0.0 # Force deterministic and factual answers
    )
    
    # 3. Create the prompt
    # Strict instructions to avoid hallucinations but allow reasonable comprehension
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful Enterprise AI assistant. Answer the user's question based strictly on the provided context. You may infer meaning from the context to answer the question, but do not hallucinate outside knowledge. If the answer cannot be reasonably found in the context, say 'I cannot answer this based on the provided documents.'\n\nCONTEXT:\n{context}"),
        ("human", "{question}")
    ])
    
    # 4. Generate the response using LECL (LangChain Expression Language)
    chain = prompt | llm
    
    print(f"[INFO] Generating answer with Llama 3 on Groq...")
    response = chain.invoke({
        "context": context_text,
        "question": query
    })
    
    print("\n================ FINAL ANSWER ================\n")
    print(response.content)
    print("\n==============================================\n")
    return response.content

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full RAG Pipeline with Groq LLM")
    parser.add_argument("query", type=str, help="The question to ask the RAG pipeline")
    parser.add_argument("--api-key", type=str, default=os.environ.get("GROQ_API_KEY"), help="Groq API Key (or set GROQ_API_KEY env var)")
    
    args = parser.parse_args()
    
    if not args.api_key:
        print("[ERROR] Please provide a Groq API Key via --api-key or set the GROQ_API_KEY environment variable.")
        exit(1)
        
    generate_answer(args.query, args.api_key)

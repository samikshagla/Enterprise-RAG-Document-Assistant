import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

# Import our RAG functions
from vector_db import store_in_chroma
from rag_pipeline import generate_answer

app = FastAPI(title="Enterprise RAG Backend")

# We assume GROQ_API_KEY is available in the environment to simplify this implementation
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

class ChatRequest(BaseModel):
    query: str

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Receives a PDF upload natively via a multipart form HTTP request,
    stores it in a temporary folder, and calls our ingestion logic.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    os.makedirs("temp_uploads", exist_ok=True)
    temp_path = os.path.join("temp_uploads", file.filename)
    
    try:
        # Write the incoming file stream to local disk
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
            
        print(f"[INFO] File temporarily saved at {temp_path}")
        
        # Trigger our Phase 3 indexing pipeline
        vectorstore = store_in_chroma(temp_path)
        
        if not vectorstore:
            raise HTTPException(status_code=500, detail="Failed to store the document into ChromaDB.")
            
        return {"status": "success", "message": f"Successfully processed and embedded {file.filename}."}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Auto-cleanup the temporary file so it doesn't clutter the server.
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"[INFO] Auto-cleaned temporary file: {temp_path}")


@app.post("/chat")
async def chat_with_docs(request: ChatRequest):
    """
    Takes a JSON payload containing the user's question, triggering
    the vector database retrieval and Llama 3 generation pipeline.
    """
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY environment variable is not configured on the backend server.")
        
    try:
        print(f"[API] Processing new chat query: '{request.query}'")
        # Run our Phase 4 function
        response_text = generate_answer(request.query, groq_api_key=GROQ_API_KEY)
        
        if not response_text:
            return {"answer": "I do not have enough context loaded to answer that question."}
            
        return {"answer": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Start the local development server (binds to localhost:8000 by default)
    print("Starting FastAPI Backend Server...")
    uvicorn.run("backend:app", host="127.0.0.1", port=8000, reload=True)

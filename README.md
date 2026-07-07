---
title: Enterprise RAG Document Assistant
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# 🤖 Enterprise RAG Document Assistant

A full-stack, AI-powered Retrieval-Augmented Generation (RAG) system running entirely inside Docker containers. This platform allows you to securely upload enterprise PDF documents, index their knowledge using localized vector embeddings, and interface with powerful LLMs (like LLaMA 3 via Groq) to accurately answer questions based **strictly** on the ingested context.

---

## 🏗️ Architecture overview

This project implements a multi-tier microservice architecture:
1. **Frontend**: A Streamlit chat UI for seamless user interaction and drag-and-drop document uploads.
2. **Backend**: A FastAPI REST interface that orchestrates the data pipeline, document chunking, embeddings generation, and retrieval logic.
3. **Database**: A local ChromaDB instance that persists multi-dimensional vector embeddings of your file chunks securely on your machine.
4. **LLM Inference**: The generation layer relies on LangChain expression components passing the synthesized context string to Groq's high-speed endpoint.

## 🛠️ Tech Stack

- **Data Parsing**: `PyMuPDF`
- **Orchestration**: `LangChain`
- **Vector Embeddings**: `SentenceTransformers` (`all-MiniLM-L6-v2`)
- **Vector Database**: `ChromaDB`
- **LLM Inferencing Engine**: `Groq` API (`llama-3.1-8b-instant`)
- **Backend API**: `FastAPI` / `Uvicorn`
- **Frontend App**: `Streamlit`
- **Deployment**: `Docker` & `Docker Compose`

---

## 🚀 Getting Started Locally

### Prerequisites
- [Docker & Docker Compose](https://www.docker.com/products/docker-desktop/) installed on your machine.
- A free API key from [Groq](https://console.groq.com/keys).

### Installation & Run

1. **Clone the repository**:
   ```bash
   git clone <YOUR-GITHUB-URL>
   cd Enterprise-RAG-Document-Assistant
   ```

2. **Set your Groq API Environment Variable**:
   For Linux/macOS:
   ```bash
   export GROQ_API_KEY="gsk_your_groq_api_key_here"
   ```
   For Windows PowerShell:
   ```powershell
   $env:GROQ_API_KEY="gsk_your_groq_api_key_here"
   ```

3. **Deploy using Docker Compose**:
   ```bash
   docker-compose up --build
   ```

*(Note: The very first build may take a few minutes as it downloads PyTorch and the HuggingFace `all-MiniLM-L6-v2` embedding model to bake natively into the Docker container cache).*

### Usage

Once the containers are successfully running:
- Open your browser to **[http://localhost:8501](http://localhost:8501)** to access the Streamlit Chat interface.
- Upload any PDF file using the left sidebar. Wait for the `Success!` alert.
- Type any question into the chat, and the Assistant will search your ingested files to formulate an answer!

---

## 🤝 Contribution Requirements
If you submit a pull request, ensure that you follow Conventional Commits formatting and have passed standard Python type checks.

**License**: MIT 

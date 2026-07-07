#!/bin/bash
# Start FastAPI backend in the background
python -m uvicorn backend:app --host 127.0.0.1 --port 8000 &

# Start Streamlit frontend in the foreground using the platform's PORT (or default 8501)
PORT=${PORT:-8501}
python -m streamlit run frontend.py --server.port $PORT --server.address 0.0.0.0

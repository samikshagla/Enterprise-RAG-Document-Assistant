FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required for building C extensions or running specific python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies first to leverage Docker cache
COPY requirements.txt .

# Install dependencies (this might take a bit for ML libraries like torch/transformers)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of our application code
COPY . .

# Expose the ports that FastAPI and Streamlit will use
EXPOSE 8000
EXPOSE 8501

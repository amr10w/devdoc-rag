# DevDoc RAG — Production Container
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (curl for healthchecks, build-essential for C extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy packaging metadata first to leverage Docker layer caching
COPY pyproject.toml README.md ./

# Upgrade pip and install package dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Copy application source code
COPY src/ /app/src/

# Environment configuration
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app" \
    API_BASE_URL="http://localhost:8000" \
    OLLAMA_API_URL="https://ollama.com" \
    OLLAMA_MODEL="glm-5.2:cloud" \
    RAG_LOGS_DB="/app/src/data/rag_logs.db"

# Expose ports: 8000 (FastAPI API), 8501 (Streamlit Dashboard)
EXPOSE 8000 8501

# Default command launches FastAPI backend
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

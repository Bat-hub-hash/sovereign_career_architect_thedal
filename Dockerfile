FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY pyproject.toml .
COPY README.md .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    httpx \
    structlog \
    pydantic \
    pydantic-settings \
    python-dotenv \
    langchain \
    langchain-groq \
    langchain-openai \
    langchain-community \
    langgraph \
    openai \
    rich \
    typer \
    requests

# Copy application code
COPY src/ ./src/

# Set Python path to include src directory
ENV PYTHONPATH=/app/src

# Expose port
EXPOSE 10000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:10000/ || exit 1

# Run the application (Render uses PORT env var)
CMD ["sh", "-c", "uvicorn src.sovereign_career_architect.api.main:app --host 0.0.0.0 --port ${PORT:-10000}"]

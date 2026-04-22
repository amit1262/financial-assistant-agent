FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install minimal system dependencies (for building Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml uv.lock* ./

# Install uv and dependencies
RUN pip install --no-cache-dir uv && \
    uv sync --frozen

# Copy application code
COPY . .

# Expose port for LangServe API
EXPOSE 8000

# Run the LangServe app with LangGraph agent
CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

# Stage 1: Builder
FROM python:3.12.13-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock* ./

RUN pip install --no-cache-dir uv && \
    uv sync --frozen

# Stage 2: Production (minimal)
FROM python:3.12.13-slim

WORKDIR /app

# Copy app code
COPY . .

# Copy compiled dependencies from builder
COPY --from=builder /app/.venv .venv

# Copy uv binary from builder
COPY --from=builder /usr/local/bin/uv /usr/local/bin/uv


EXPOSE 8000

ENV PATH="/app/.venv/bin:$PATH"
CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
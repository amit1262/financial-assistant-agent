FROM ghcr.io/astral-sh/uv:latest AS uv_bin

FROM python:3.13.13-slim

# Copy the uv binary into your path
COPY --from=uv_bin /uv /usr/local/bin/uv

# Set optimization environment variables
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

WORKDIR /app

# Copy everything to the container
COPY . .

# install dependencies and build the virtual environment
RUN uv lock && \
    uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

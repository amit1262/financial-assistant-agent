# Use the official Ollama image with a stable version
FROM ollama/ollama:0.15.0

# Expose the default Ollama port
EXPOSE 11434

# Create and set working directory
WORKDIR /app
RUN mkdir -p /app

# Pre-download llama models suitable for Mac (smaller, efficient models)
# These models are optimized for systems with limited resources
RUN ollama serve & \
    sleep 5 && \
    ollama pull llama3.2:3b && \
    # ollama pull llama3.2:8b && \
    pkill ollama

# Set the entrypoint
ENTRYPOINT ["/bin/ollama"]

# Default command
CMD ["serve"]

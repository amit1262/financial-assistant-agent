"""
Initialize and provide Ollama LLM instance.
"""

import os
import logging
from langchain_ollama import ChatOllama
from langchain_ollama._utils import validate_model
from ollama import Client

logger = logging.getLogger(__name__)


# Initialize model once at module load time
def _initialize_model():
    """Initialize Ollama Chat LLM from environment variables"""
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    model_name = os.getenv("MODEL_NAME", "llama2")

    logger.info(f"Initializing Ollama model: {model_name} at {ollama_base_url}")

    try:
        # Create Ollama client and validate model
        client = Client(host=ollama_base_url)
        validate_model(client, model_name)
        logger.info(f"Model '{model_name}' validated successfully")

        # Initialize Chat LLM
        llm = ChatOllama(model=model_name, base_url=ollama_base_url)
        logger.info(f"Connected to Ollama at {ollama_base_url} with model {model_name}")

        return llm

    except ValueError as ve:
        logger.error(f"Model validation error for '{model_name}': {ve}")
        raise RuntimeError(f"Model validation failed: {ve}")

    except Exception as e:
        logger.error(f"Failed to initialize model '{model_name}': {e}")
        raise RuntimeError(f"Error initializing Ollama model: {e}")


# Singleton instance - created once when module is imported
language_model = _initialize_model()

"""
Initialize and provide OpenRouter LLM instance.
"""

import os
from langchain_openrouter import ChatOpenRouter
import logging

logger = logging.getLogger(__name__)


# Initialize model once at module load time
def _initialize_model():
    """Initialize Chat LLM from environment variables"""
    model_name = os.getenv("MODEL_NAME")
    if model_name is None:
        logger.error("MODEL_NAME environment variable is not set")
        raise RuntimeError("MODEL_NAME environment variable is required")

    logger.info(f"Initializing model: {model_name}")

    try:
        model = ChatOpenRouter(
            model=model_name,
            # temperature=0,
            # max_tokens=1024,
            # max_retries=2,
        )
        logger.info(f"Initialized model '{model_name}' successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize model '{model_name}': {e}")
        raise RuntimeError(f"Error initializing model: {e}")


language_model = _initialize_model()

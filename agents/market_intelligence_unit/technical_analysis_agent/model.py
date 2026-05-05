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
    model_name = os.getenv("TECHNICAL_ANALYSIS_MODEL_NAME")
    if model_name is None:
        logger.error("TECHNICAL_ANALYSIS_MODEL_NAME environment variable is not set")
        raise RuntimeError(
            "TECHNICAL_ANALYSIS_MODEL_NAME environment variable is required"
        )

    try:
        model = ChatOpenRouter(
            model=model_name,
            temperature=float(os.getenv("TECHNICAL_ANALYSIS_MODEL_TEMPERATURE", 0.2)),
            # max_tokens=1024,
            max_retries=5,
        )
        logger.info(f"Initialized model: {model_name} ")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize model '{model_name}': {e}")
        raise RuntimeError(f"Error initializing model: {e}")


model = _initialize_model()

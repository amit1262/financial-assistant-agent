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
    model_name = os.getenv("NEWS_ANALYSIS_MODEL_NAME")
    if model_name is None:
        logger.error("NEWS_ANALYSIS_MODEL_NAME environment variable is not set")
        raise RuntimeError("NEWS_ANALYSIS_MODEL_NAME environment variable is required")

    try:
        model = ChatOpenRouter(
            model=model_name,
            temperature=float(os.getenv("NEWS_ANALYSIS_MODEL_TEMPERATURE", 0.2)),
            # max_tokens=1024,
            max_retries=5,
        )
        logger.info(f"[News Agent] Initialized model: {model_name} ")
        return model
    except Exception as e:
        logger.error(f"[News Agent] Failed to initialize model '{model_name}': {e}")
        raise RuntimeError(f"[News Agent] Error initializing model: {e}")


model = _initialize_model()

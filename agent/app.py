from fastapi import FastAPI
import logging
import sys

# Setup logging BEFORE any other imports (force=True overrides existing config)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)
logger.info("=" * 50)
logger.info("Starting Financial Assistant Agent")
logger.info("=" * 50)

from routers.query_router import router as query_router

logger.info("Query router loaded successfully")

# Initialize FastAPI
app = FastAPI()
# Setup routes
app.include_router(query_router)

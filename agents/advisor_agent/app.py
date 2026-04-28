import logging
import sys
import os
from fastapi import FastAPI
import mlflow

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

# import routers after logging is configured to ensure any logs from routers are captured
from routers.query_router import router as query_router


logger.info("Query router loaded successfully")

# setup mlflow tracking
mlflow.langchain.autolog()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME"))
logger.info(
    f"MLflow tracking URI: {os.getenv('MLFLOW_TRACKING_URI')}, "
    f"experiment: {os.getenv('MLFLOW_EXPERIMENT_NAME')}"
)

# Initialize FastAPI
app = FastAPI()
# Setup routes
app.include_router(query_router)

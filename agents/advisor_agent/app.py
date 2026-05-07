import logging
import sys
import os
from fastapi import FastAPI
import mlflow
from contextlib import asynccontextmanager

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

# imports after logging is configured to ensure any logs from routers/services are captured
from routers.query_router import router as query_router
from services.agent_manager import manager
import src.agent as advisor_agent

logger.info("Query router loaded successfully")


@asynccontextmanager
async def lifespan(app: FastAPI):
    "Lifespan function to initialize resources before the app starts accepting requests"
    try:
        # Initialize MLflow tracking inside the lifespan to ensure correct async context
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME"))
        mlflow.langchain.autolog()
        logger.info(
            f"MLflow Tracking URI: {os.getenv('MLFLOW_TRACKING_URI')}, "
            f"Experiment Name: {os.getenv('MLFLOW_EXPERIMENT_NAME')} initialized in lifespan"
        )

        # create agent graph and register with manager
        advisor_agent_graph = await advisor_agent.build_agent_graph()
        if advisor_agent_graph is None:
            logger.error(
                "Failed to build advisor agent graph. Check previous logs for details."
            )
            raise Exception("Advisor agent graph construction failed.")
        manager.register_agent(agent_name="advisor", agent_instance=advisor_agent_graph)

        yield  # app runs after this point

        # cleanup models before shutdown
        logger.info("Shutting down Financial Assistant Agent")
        manager.agents.clear()

    except Exception as e:
        logger.error(
            f"[App Lifespan] Unexpected error during startup/shutdown: {e}",
            exc_info=True,
        )


# Initialize FastAPI
app = FastAPI(lifespan=lifespan)
# Setup routes
app.include_router(router=query_router)

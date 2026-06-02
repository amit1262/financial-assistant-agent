from contextlib import asynccontextmanager
import logging
import sys
import os
from fastapi import FastAPI
import mlflow

# sub-agents imports
from technical_analysis_agent.agent_server import TechnicalAnalysisAgentServer
from fundamental_analysis_agent.agent_server import FundamentalAnalysisAgentServer
from news_analysis_agent.agent_server import NewsAnalysisAgentServer

# Setup logging BEFORE any other imports (force=True overrides existing config)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)
logger.info("=" * 50)
logger.info("Starting Market Intelligence Unit of Agents")
logger.info("=" * 50)


@asynccontextmanager
async def lifespan(app: FastAPI):
    "Lifespan function to initialize resources before the app starts accepting requests"
    try:
        # Initialize MLflow tracking inside the lifespan to ensure correct async context
        # mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        # mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME"))
        # mlflow.langchain.autolog()
        # logger.info(
        #     f"MLflow Tracking URI: {os.getenv('MLFLOW_TRACKING_URI')}, "
        #     f"Experiment Name: {os.getenv('MLFLOW_EXPERIMENT_NAME')} initialized in lifespan"
        # )

        # initialize agent servers (cascaded loading of agent, fastapi app, tools, model, graph)
        await technical_agent_server.initialize()
        await fundamental_agent_server.initialize()
        await news_agent_server.initialize()

        # startup information display
        logger.info("Agents initialized and registered.")
        logger.info("-" * 50)
        logger.info("Technical Analysis Agent Started Successfully.")
        logger.info("Fundamental Analysis Agent Started Successfully.")
        logger.info("News Analysis Agent Started Successfully.")
        logger.info("-" * 50)

        yield  # app runs after this point
        # cleanup models before shutdown
        logger.info("Shutting down Market Intelligence Unit of Agents")
        mlflow.end_run()  # ensure MLflow run is ended on shutdown

    except Exception as e:
        logger.error(f"Error during app lifespan: {e}", exc_info=True)
        raise e  # re-raise to prevent app from starting in a bad state


# Initialize FastAPI
app = FastAPI(lifespan=lifespan)

# Initialize and register sub-agents (technical, fundamental, news)
technical_agent_server = TechnicalAnalysisAgentServer()
fundamental_agent_server = FundamentalAnalysisAgentServer()
news_agent_server = NewsAnalysisAgentServer()

# mount sub-agents to main app (mount the FastAPI subapi, not the server object)
app.mount("/technical", technical_agent_server.subapp)
app.mount("/fundamental", fundamental_agent_server.subapp)
app.mount("/news", news_agent_server.subapp)
logger.info(
    "Market Intelligence Unit setup complete. FastAPI app is ready to serve requests."
)

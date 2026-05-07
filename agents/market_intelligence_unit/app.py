from contextlib import asynccontextmanager
import logging
import sys
import os
from fastapi import FastAPI
import mlflow
from routers.query_router import router
from services.agent_manager import manager as agent_manager
from services.mcp_client_manager import mcp_manager

# sub-agents imports
import technical_analysis_agent.agent as technical_agent
import fundamental_analysis_agent.agent as fundamental_agent
import news_analysis_agent.agent as news_agent
from technical_analysis_agent.mcp_client import MCPClient as TechMCPClient
from fundamental_analysis_agent.mcp_client import MCPClient as FundamentalMCPClient
from news_analysis_agent.mcp_client import MCPClient as NewsMCPClient


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
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME"))
        mlflow.langchain.autolog()
        logger.info(
            f"MLflow Tracking URI: {os.getenv('MLFLOW_TRACKING_URI')}, "
            f"Experiment Name: {os.getenv('MLFLOW_EXPERIMENT_NAME')} initialized in lifespan"
        )

        # technical agent graph construction and mcp client setup
        tech_mcp_client = await TechMCPClient.create()
        mcp_manager.register_client("technical", tech_mcp_client)
        tech_agent = await technical_agent.build_agent_graph()
        agent_manager.register_agent(
            agent_name="technical",
            agent_instance=tech_agent,
        )
        # fundamental analysis agent graph construction and mcp client setup
        fundamental_mcp_client = await FundamentalMCPClient.create()
        mcp_manager.register_client("fundamental", fundamental_mcp_client)
        fundamental_agent_instance = await fundamental_agent.build_agent_graph()
        agent_manager.register_agent(
            agent_name="fundamental",
            agent_instance=fundamental_agent_instance,
        )

        # news analysis agent graph construction and mcp client setup
        news_mcp_client = await NewsMCPClient.create()
        mcp_manager.register_client("news", news_mcp_client)
        news_agent_instance = await news_agent.build_agent_graph()
        agent_manager.register_agent(
            agent_name="news",
            agent_instance=news_agent_instance,
        )

        # startup information display
        logger.info("Agent Manager initialized and agents registered.")
        logger.info("-" * 50)
        logger.info("Technical Analysis Agent Started Successfully.")
        logger.info("Fundamental Analysis Agent Started Successfully.")
        logger.info("News Analysis Agent Started Successfully.")
        logger.info("-" * 50)

        yield  # app runs after this point
        # cleanup models before shutdown
        logger.info("Shutting down Market Intelligence Unit of Agents")
        agent_manager.agents.clear()
        mcp_manager.clients.clear()
        mlflow.end_run()  # ensure MLflow run is ended on shutdown

    except Exception as e:
        logger.error(f"Error during app lifespan: {e}", exc_info=True)
        raise e  # re-raise to prevent app from starting in a bad state


# Initialize FastAPI
app = FastAPI(lifespan=lifespan)
# Setup routes
app.include_router(router)

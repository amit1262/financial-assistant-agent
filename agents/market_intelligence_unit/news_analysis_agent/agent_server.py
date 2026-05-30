# Agent Server class (A2AStarletteApplication), which initializes the agent executor, and serves the agent card metadata
import logging
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, AgentInterface
from news_analysis_agent.agent_executor import NewsAnalysisAgentExecutor
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
import os
from fastapi import FastAPI

logger = logging.getLogger(__name__)


class NewsAnalysisAgentServer:
    def __init__(self):
        self.host = "market_intelligence_service"  # service name for docker network
        self.port = os.getenv("MIU_PORT", "8001")
        self.routes = []
        self.subapp = FastAPI()

    async def initialize(self):
        # Define agent skills
        skills = [
            AgentSkill(
                id="news-data-retrieval",
                name="News Analysis Data Retrieval",
                description=(
                    "Retrieves and analyzes news data for specified companies and sectors. "
                    "Executes precise tool calls to fetch news articles, sentiment analysis, and market impact "
                    "information. Works as a specialized sub-agent that translates "
                    "news analysis queries from the main orchestrator into accurate tool calls and verifies news data. "
                    "Focuses exclusively on news analysis without providing subjective investment advice."
                ),
                tags=[
                    "news-analysis",
                    "sentiment-analysis",
                    "market-impact",
                    "news-data",
                    "sub-agent",
                ],
                examples=[
                    "Fetch latest news articles for AAPL",
                    "Analyze sentiment of news articles for Tesla",
                    "Assess market impact of recent news for Microsoft",
                ],
                input_modes=["text/plain"],
                output_modes=["text/plain"],
            )
        ]
        # Create agent card
        agent_card = AgentCard(
            name="News Analysis Agent",
            description=(
                "A specialized news analysis sub-agent that serves as part of a multi-agent financial analysis system. "
                "Receives news analysis queries from the main orchestrator agent and translates them into precise tool calls "
                "to retrieve and verify news data. Leverages MCP-based news analysis tools to extract metrics such as sentiment, market impact, and news trends for any ticker and industry. Returns data-focused news analysis results without subjective investment advice. "
                "Handles failures gracefully by reporting specific errors and supports comprehensive news data retrieval "
                "(latest articles, historical trends, sentiment analysis)."
            ),
            version="1.0",
            capabilities=AgentCapabilities(streaming=True),
            skills=skills,
            default_input_modes=["text/plain"],
            default_output_modes=["text/plain"],
            supported_interfaces=[
                AgentInterface(
                    protocol_binding="JSONRPC",
                    url=f"http://{self.host}:{self.port}/news/jsonrpc",
                )
            ],
        )
        # create request handler and A2A Starlette Application
        request_handler = DefaultRequestHandler(
            agent_executor=await NewsAnalysisAgentExecutor().initialize(),
            task_store=InMemoryTaskStore(),
            agent_card=agent_card,
        )
        self.subapp.router.routes.extend(create_agent_card_routes(agent_card))
        self.subapp.router.routes.extend(
            create_jsonrpc_routes(request_handler, rpc_url="/news/jsonrpc")
        )

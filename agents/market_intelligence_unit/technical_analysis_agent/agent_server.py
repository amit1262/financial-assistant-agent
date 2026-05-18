# Agent Server class (A2AStarletteApplication), which initializes the agent executor, and serves the agent card metadata
import logging
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, AgentInterface
from technical_analysis_agent.agent_executor import TechnicalAnalysisAgentExecutor
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
import os
from fastapi import FastAPI

logger = logging.getLogger(__name__)


class TechnicalAnalysisAgentServer:
    def __init__(self):
        self.host = "market_intelligence_service"  # service name for docker network
        self.port = os.getenv("MIU_PORT", "8001")
        self.routes = []
        self.subapp = FastAPI()

    async def initialize(self):
        # Define agent skills
        skills = [
            AgentSkill(
                id="technical-data-retrieval",
                name="Technical Analysis Data Retrieval",
                description=(
                    "Retrieves and analyzes technical market data for specified tickers and timeframes. "
                    "Executes precise tool calls to fetch historical price-action data and compute technical indicators "
                    "(SMA, EMA, RSI, MACD, Bollinger Bands, etc.). Works as a specialized sub-agent that translates "
                    "technical analysis queries from the main orchestrator into accurate tool calls and verifies market data. "
                    "Focuses exclusively on price-action data retrieval without providing subjective investment advice."
                ),
                tags=[
                    "technical-indicators",
                    "market-data",
                    "price-action",
                    "sma",
                    "ema",
                    "rsi",
                    "macd",
                    "bollinger-bands",
                    "data-retrieval",
                    "sub-agent",
                ],
                examples=[
                    "Fetch SMA(20) and EMA(50) for AAPL on daily timeframe for the last 100 days",
                    "Calculate RSI(14) and MACD for BTC on hourly timeframe",
                    "Get Bollinger Bands(20, 2) for SPY daily chart",
                ],
                input_modes=["text/plain"],
                output_modes=["text/plain"],
            )
        ]
        # Create agent card
        agent_card = AgentCard(
            name="Technical Analysis Agent",
            description=(
                "A specialized technical analysis sub-agent that serves as part of a multi-agent financial analysis system. "
                "Receives technical analysis queries from the main orchestrator agent and translates them into precise tool calls "
                "to retrieve and verify market data. Leverages MCP-based technical analysis tools to compute indicators such as SMA, EMA, RSI, MACD, "
                "and Bollinger Bands for any ticker and timeframe. Returns data-focused analysis results without subjective investment advice. "
                "Handles failures gracefully by reporting specific errors and supports configurable data intervals (default: daily) and data point ranges "
                "(default: last 100 points for sufficient cross-over analysis)."
            ),
            version="1.0",
            capabilities=AgentCapabilities(streaming=True),
            skills=skills,
            default_input_modes=["text/plain"],
            default_output_modes=["text/plain"],
            supported_interfaces=[
                AgentInterface(
                    protocol_binding="JSONRPC",
                    url=f"http://{self.host}:{self.port}/technical/jsonrpc",
                )
            ],
        )
        # create request handler and A2A Starlette Application
        request_handler = DefaultRequestHandler(
            agent_executor=await TechnicalAnalysisAgentExecutor().initialize(),
            task_store=InMemoryTaskStore(),
            agent_card=agent_card,
        )
        self.subapp.router.routes.extend(create_agent_card_routes(agent_card))
        self.subapp.router.routes.extend(
            create_jsonrpc_routes(request_handler, rpc_url="/jsonrpc")
        )

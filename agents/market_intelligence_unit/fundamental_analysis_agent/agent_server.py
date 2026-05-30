# Agent Server class (A2AStarletteApplication), which initializes the agent executor, and serves the agent card metadata
import logging
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, AgentInterface
from fundamental_analysis_agent.agent_executor import FundamentalAnalysisAgentExecutor
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
import os
from fastapi import FastAPI

logger = logging.getLogger(__name__)


class FundamentalAnalysisAgentServer:
    def __init__(self):
        self.host = "market_intelligence_service"  # service name for docker network
        self.port = os.getenv("MIU_PORT", "8001")
        self.routes = []
        self.subapp = FastAPI()

    async def initialize(self):
        # Define agent skills
        skills = [
            AgentSkill(
                id="fundamental-data-retrieval",
                name="Fundamental Analysis Data Retrieval",
                description=(
                    "Retrieves and analyzes fundamental financial data for specified companies and sectors. "
                    "Executes precise tool calls to fetch financial statements, valuation metrics, and company fundamentals "
                    "(P/E ratio, EPS, earnings, revenue, balance sheet, cash flow, debt ratios, ROE, etc.). Works as a specialized sub-agent that translates "
                    "fundamental analysis queries from the main orchestrator into accurate tool calls and verifies financial data. "
                    "Focuses exclusively on financial statement analysis and valuation metrics without providing subjective investment advice."
                ),
                tags=[
                    "fundamental-analysis",
                    "financial-statements",
                    "valuation",
                    "earnings",
                    "revenue",
                    "balance-sheet",
                    "cash-flow",
                    "ratios",
                    "financial-data",
                    "sub-agent",
                ],
                examples=[
                    "Fetch P/E ratio, EPS, and earnings history for AAPL",
                    "Calculate debt-to-equity ratio and ROE for Tesla",
                    "Get revenue trends, operating margins, and cash flow statements for Microsoft",
                ],
                input_modes=["text/plain"],
                output_modes=["text/plain"],
            )
        ]
        # Create agent card
        agent_card = AgentCard(
            name="Fundamental Analysis Agent",
            description=(
                "A specialized fundamental analysis sub-agent that serves as part of a multi-agent financial analysis system. "
                "Receives fundamental analysis queries from the main orchestrator agent and translates them into precise tool calls "
                "to retrieve and verify financial data. Leverages MCP-based fundamental analysis tools to extract metrics such as P/E ratio, earnings, revenue, "
                "balance sheet data, cash flows, and valuation ratios for any ticker and industry. Returns data-focused financial analysis results without subjective investment advice. "
                "Handles failures gracefully by reporting specific errors and supports comprehensive financial data retrieval "
                "(quarterly, annual reports, historical trends)."
            ),
            version="1.0",
            capabilities=AgentCapabilities(streaming=True),
            skills=skills,
            default_input_modes=["text/plain"],
            default_output_modes=["text/plain"],
            supported_interfaces=[
                AgentInterface(
                    protocol_binding="JSONRPC",
                    url=f"http://{self.host}:{self.port}/fundamental/jsonrpc",
                )
            ],
        )
        # create request handler and A2A Starlette Application
        request_handler = DefaultRequestHandler(
            agent_executor=await FundamentalAnalysisAgentExecutor().initialize(),
            task_store=InMemoryTaskStore(),
            agent_card=agent_card,
        )
        self.subapp.router.routes.extend(create_agent_card_routes(agent_card))
        self.subapp.router.routes.extend(
            create_jsonrpc_routes(request_handler, rpc_url="/fundamental/jsonrpc")
        )

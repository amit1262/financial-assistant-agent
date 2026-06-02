# class that defines and implements news analysis agent for market intelligence unit

"agent state graph definition."

"To Do - potential node additions for future iterations - "
"1. Add thought trace extraction node after generator to capture intermediate reasoning steps"
"2. if same tool calls are made in a loop, add detection and loop breaking mechanism to avoid infinite cycles. Use State."
"3. Add heartbeat to monitor MCP servers."


import os
from typing import Any
import logging
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openrouter import ChatOpenRouter
from functools import partial
from langchain_core.runnables import Runnable
from news_analysis_agent.mcp_client import MCPClient
from news_analysis_agent.state import State
from news_analysis_agent.graph_nodes.generator import generator
from news_analysis_agent.graph_nodes.tool_checker import tool_checker


logger = logging.getLogger(__name__)


class NewsAnalysisAgent:
    def __init__(self):
        self.mcp_tools: list = None
        self.graph = None
        self.model: Runnable = None

    async def initialize(self):
        # initialize mcp client and fetch tools
        await self._get_mcp_tools()
        # initialize model with tool binding
        await self._initialize_model()
        # initialize agent graph
        await self._build_graph()

    # LLM used by the agent
    async def _initialize_model(self):
        """Initialize Chat LLM from environment variables"""
        model_name = os.getenv("NEWS_MODEL_NAME")
        base_url = os.getenv("BASE_URL")
        api_key = os.getenv("OPENROUTER_API_KEY")
        if model_name is None:
            logger.error("NEWS_MODEL_NAME environment variable is not set")
            raise RuntimeError("NEWS_MODEL_NAME environment variable is required")
        if base_url is None:
            logger.error("BASE_URL environment variable is not set")
            raise RuntimeError("BASE_URL environment variable is required")
        if api_key is None:
            logger.error("OPENROUTER_API_KEY environment variable is not set")
            raise RuntimeError("OPENROUTER_API_KEY environment variable is required")

        try:
            model = ChatOpenRouter(
                model=model_name,
                base_url=base_url,
                api_key=api_key,
            )
            self.model = model
            logger.info(f"[News Agent] Initialized model: {model_name}")
        except Exception as e:
            logger.error(f"[News Agent] Failed to initialize model '{model_name}': {e}")
            raise RuntimeError(f"[News Agent] Error initializing model: {e}")

    async def _build_graph(self) -> None:
        # build and return the agent's workflow graph using LangGraph
        # workflow graph definition
        workflow = StateGraph(state_schema=State)

        # add graph nodes
        tool_checker_partial = partial(
            tool_checker, mcp_tools=self.mcp_tools
        )  # inject tools into tool_checker node
        workflow.add_node("tool_checker", tool_checker_partial)
        generator_partial = partial(
            generator, model=self.model
        )  # inject model into generator node
        workflow.add_node("generator", generator_partial)
        workflow.add_node("tool_node", ToolNode(tools=self.mcp_tools))

        # add edges
        workflow.add_edge(
            START, "tool_checker"
        )  # can either goto END or generator based on tool_checker output
        workflow.add_conditional_edges(
            "generator", tools_condition, {"tools": "tool_node", "__end__": END}
        )
        workflow.add_edge("tool_node", "generator")
        # state persistence setup
        memory = MemorySaver()
        # compile graph
        graph = workflow.compile(checkpointer=memory)

        self.graph = graph

    async def _get_mcp_tools(self) -> None:
        # create and return the MCP client for this agent
        mcp_client = await MCPClient().initialize()
        self.mcp_tools = await mcp_client.get_mcp_tools()
        logger.info(
            f"[News Agent] Fetched MCP tools: {[tool.name for tool in self.mcp_tools]}"
        )

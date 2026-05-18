# class that defines and implements advisor agent for financial analysis assistant application

from functools import partial
import os

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from src.state import State
from src.graph_nodes.synthesizer import synthesizer
from src.graph_nodes.agent_card_retriver import cards_retriever
from src.graph_nodes.tool_node import tool_executor
from src.graph_nodes.planner import planner
import logging
from langchain_openrouter import ChatOpenRouter
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)


class AdvisorAgent:
    def __init__(self):
        self.graph = None
        self.model = None

    async def initialize(self):
        # initialize model
        await self._initialize_model()
        # initialize agent graph
        await self._build_graph()
        return self

    # LLM used by the agent
    async def _initialize_model(self):
        """Initialize Chat LLM from environment variables"""
        model_name = os.getenv("ADVISOR_MODEL_NAME")
        if model_name is None:
            logger.error("ADVISOR_MODEL_NAME environment variable is not set")
            raise RuntimeError("ADVISOR_MODEL_NAME environment variable is required")
        try:
            # model = ChatOpenRouter(
            #     model=model_name,
            #     temperature=float(os.getenv("ADVISOR_MODEL_TEMPERATURE", 0.2)),
            #     # max_tokens=1024,
            #     max_retries=5,
            # )
            model = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=1.0,
                max_retries=2,
                google_api_key=os.getenv("GEMINI_API_KEY"),
            )
            self.model = model
            logger.info(f"[Advisor Agent] Initialized model: {model_name}")
        except Exception as e:
            logger.error(
                f"[Advisor Agent] Failed to initialize model '{model_name}': {e}"
            )
            raise RuntimeError(f"[Advisor Agent] Error initializing model: {e}")

    # Build LangGraph workflow

    async def _build_graph(self):
        # workflow graph definition
        workflow = StateGraph(state_schema=State)

        # add graph nodes
        workflow.add_node("agent_card_retriever", cards_retriever)
        planner_node = partial(planner, model=self.model)
        workflow.add_node("planner", planner_node)
        workflow.add_node("tool_executor", tool_executor)
        synthesizer_node = partial(synthesizer, model=self.model)
        workflow.add_node("synthesizer", synthesizer_node)

        # add edges
        workflow.add_edge(START, "agent_card_retriever")
        workflow.add_edge(
            "agent_card_retriever", "planner"
        )  # planner determines the next node based on the state and feedback
        workflow.add_edge(
            "tool_executor", "synthesizer"
        )  # synthesizer determines the next node based on completeness of data and feedback

        # state persistence setup
        memory = MemorySaver()
        # compile graph
        graph = workflow.compile(checkpointer=memory)
        self.graph = graph
        logger.info("[Advisor Agent] Agent graph built and compiled successfully")

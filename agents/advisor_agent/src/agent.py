# class that defines and implements advisor agent for financial analysis assistant application

from functools import partial
import os
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from src.state import State
from src.graph_nodes.synthesizer import synthesizer, synthesizer_condition
from src.graph_nodes.agent_card_retriver import cards_retriever
from src.graph_nodes.sub_agent_executor import sub_agent_executor
from src.graph_nodes.planner import planner, planner_condition
import logging
from langchain_openrouter import ChatOpenRouter

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
        base_url = os.getenv("BASE_URL")
        api_key = os.getenv("OPENROUTER_API_KEY")
        if model_name is None:
            logger.error("ADVISOR_MODEL_NAME environment variable is not set")
            raise RuntimeError("ADVISOR_MODEL_NAME environment variable is required")
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
                max_retries=5,
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
        workflow.add_node("sub_agent_executor", sub_agent_executor)
        synthesizer_node = partial(synthesizer, model=self.model)
        workflow.add_node("synthesizer", synthesizer_node)

        # add edges
        workflow.add_edge(START, "agent_card_retriever")
        # planner determines the next node based on the state and feedback
        workflow.add_edge("agent_card_retriever", "planner")
        workflow.add_conditional_edges(
            "planner",
            planner_condition,
            {
                "sub_agents": "sub_agent_executor",
                "no_sub_agents": "synthesizer",
                "end": END,
            },
        )
        # synthesizer determines the next node based on completeness of data and feedback
        workflow.add_edge("sub_agent_executor", "synthesizer")
        workflow.add_conditional_edges(
            "synthesizer",
            synthesizer_condition,
            {"planner": "planner", "end": END},
        )

        # state persistence setup
        memory = MemorySaver()
        # compile graph
        graph = workflow.compile(checkpointer=memory)
        self.graph = graph
        logger.info("[Advisor Agent] Agent graph built and compiled successfully")

        # visual graph export
        graph_image = graph.get_graph().draw_mermaid_png()
        with open("advisor_agent_graph.png", "wb") as f:
            f.write(graph_image)
        logger.info(
            "[Advisor Agent] Agent graph visualization saved as advisor_agent_graph.png"
        )

# given a user query (or feedback from synthesizer), this node plans what to do next i.e., which agents to call with what query

from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.types import Command
from langgraph.graph import END
from src.state import State
import uuid
import logging
from langchain_core.runnables import Runnable

logger = logging.getLogger(__name__)


# Define the structured output for the planner
class AgentCall(BaseModel):
    agent_name: Literal["technical", "fundamental", "news"] = Field(
        description="Name of the agent to call"
    )
    agent_query: str = Field(
        description="The specific query string to send to the sub-agent"
    )
    call_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="An id to track this agent call",
    )


class PlannerOutput(BaseModel):
    reasoning: str = Field(
        description="Short reasoning about the plan/chain-of-thoughts you have for answering the query."
    )
    agent_calls: Optional[List[AgentCall]] = Field(
        default_factory=list, description="List of sub-agent calls to make"
    )


def build_system_prompt(agent_cards: dict) -> str:
    """Build system prompt using agent card descriptions instead of hardcoded text."""

    if not agent_cards:
        agents_context = "No agents currently available.\n"
    else:
        # Extract agent capabilities from agent cards
        agents_context = ""
        for agent_name, card in agent_cards.items():
            name = card.get("name", agent_name)
            description = card.get("description", "")
            agents_context += f"- {name}: {description}\n\n"
    return (
        "You are a Financial Planning Agent. "
        "Your job is to analyze the user's query and the current state to decide which specialized sub-agents need to be consulted.\n\n"
        "AVAILABLE AGENTS:\n"
        f"{agents_context}\n"
        "GUIDELINES:\n"
        "- You can call up to 3 agents in one step.\n"
        "- When specifying agent names in agent_calls, use the internal names shown in parentheses (e.g., 'technical', 'fundamental', 'news'), NOT the full names.\n"
        "- Analyze any previous 'ToolMessages' to see what data has already been fetched.\n"
        "- If 'feedback' from the synthesizer is present in the state, only address the gaps it highlights.\n"
        "- If no more agent calls are needed to fulfill the user query, set next_node to 'synthesizer'.\n"
        "- If agent calls are required, set next_node to 'tool_executor'."
    )


async def planner(state: State, model: Runnable) -> Command:
    "Planner node: Decides whether to call sub-agents or proceed to synthesis."

    messages = state.get("messages", [])
    agent_cards = state.get("agent_cards", {})

    # If no agents are available, end the workflow
    if not agent_cards:
        logger.warning("[Planner] No agent cards available. Ending workflow.")
        return {
            "final_response": "No external agents available to process this request. Therefore, I am unable to provide an answer at this time. Please try again later when the necessary agents are available.",
            "messages": [
                AIMessage(
                    content="No external agents available. Unable to proceed. Ending workflow."
                )
            ],
            "response_complete": True,
        }

    # Build system prompt from actual agent card descriptions
    system_prompt = build_system_prompt(agent_cards)
    try:
        # Initialize the model with structured output
        planner_llm = model.with_structured_output(
            PlannerOutput, method="json_schema", strict=True
        )
        prompt = [SystemMessage(content=system_prompt)]
        prompt.extend(messages)
        response = await planner_llm.ainvoke(prompt)

        logger.info(f"[Planner] response: {response.reasoning[:100]}")
        agent_calls = response.agent_calls or []

        # Ensure each agent call has a unique ID
        for call in agent_calls:
            if not call.call_id or call.call_id == "An id to track this agent call":
                call.call_id = str(uuid.uuid4())

        return {
            "messages": [
                AIMessage(
                    content=f"[Planner response]: {response.reasoning}",
                    agent_calls=agent_calls,
                )
            ],
            "response_complete": False,
        }
    except Exception as e:
        # Fallback to synthesizer on error
        logger.error(f"Error in planner node: {e}", exc_info=True)
        return Command(
            goto="synthesizer",
            update={
                "messages": [
                    AIMessage(
                        content=f"Error in planning, falling back to synthesis: {str(e)}"
                    )
                ]
            },
        )


# Condition function for planner node to determine next step based on planner output.
def planner_condition(state: State) -> str:
    "Condition function for planner node to determine next step based on planner output."
    messages = state.get("messages", [])
    last_message = messages[-1]
    if state.get("response_complete"):
        return "end"  # If planner indicates response is complete, end the workflow
    if (
        isinstance(last_message, AIMessage)
        and hasattr(last_message, "agent_calls")
        and last_message.agent_calls
    ):
        return "sub_agents"  # If planner suggests agent calls, go to sub-agent executor
    else:
        return "no_sub_agents"  # No agent calls, go directly to synthesizer

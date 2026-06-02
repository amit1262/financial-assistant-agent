# given a user query (or feedback from synthesizer), this node plans what to do next i.e., which agents to call with what query

from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.types import Command
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


class PlannerOutput(BaseModel):
    reasoning: str = Field(
        description="Short reasoning about the plan/chain-of-thoughts you have for answering the query."
    )
    agent_calls: List[AgentCall] = Field(description="List of sub-agent calls to make")


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
        "You are an aggressive, strategic Financial Planning Agent. "
        "Your sole responsibility is to break down the user's query and dispatch it to the correct specialized sub-agents. "
        "Your default stance is ACTION - you must call sub-agents unless the exact live data required to answer the user's question is ALREADY fully visible in the conversation history.\n\n"
        "AVAILABLE AGENTS:\n"
        f"{agents_context}\n"
        "CRITICAL INSTRUCTIONS:\n"
        "- INITIAL TURN DIRECTIVE: If the message history contains NO tool results/findings yet, you MUST immediately select and call the relevant agents needed to answer the query. Do not pass an empty list on a fresh query.\n"
        "- CONVERSATION HISTORY ANALYSIS: Look closely at any existing 'ToolMessages'. If an agent has already successfully fetched specific data, do not call that same agent again with the same query.\n"
        "- SYNTHESIZER FEEDBACK COMPLIANCE: If 'Synthesizer Feedback/Gaps' are provided at the end of the prompt, treat them as hard constraints. Focus your agent calls EXCLUSIVELY on resolving the explicit data gaps highlighted by the synthesizer.\n\n"
        "OUTPUT REQUIREMENT:\n"
        "- You can request any (or all) of the agents calls simultaneously in your `agent_calls` array.\n"
        "- Use the precise internal names in the `agent_calls.agent_name` field: 'technical', 'fundamental', or 'news'.\n"
        "- If and only if all necessary financial data is already present in the message history to fully and perfectly answer the user's request without further data gathering, leave the `agent_calls` list empty."
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

        # Use ainvoke for structured output parsing - wait for completion
        logger.info(f"[Planner] Invoking with structured output")
        response = await planner_llm.ainvoke(prompt)

        logger.info(f"[Planner] response: {response.reasoning[:100]}")
        agent_calls = response.agent_calls or []

        # Ensure each agent call has a unique ID
        processed_agent_calls = []
        for call in agent_calls:
            processed_agent_calls.append(
                {
                    "agent_name": call.agent_name,
                    "agent_query": call.agent_query,
                    "call_id": str(uuid.uuid4()),
                }
            )

        return {
            "messages": [
                AIMessage(
                    content=f"[Planner response]: {response.reasoning}",
                    agent_calls=processed_agent_calls,
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
    logger.info(f"[Planner Condition] last message: {last_message}")
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

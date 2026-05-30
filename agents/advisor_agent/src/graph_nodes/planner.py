# given a user query (or feedback from synthesizer), this node plans what to do next i.e., which agents to call with what query

from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    ToolCall,
)
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
    agent_calls: Optional[List[AgentCall]] = Field(
        default_factory=list, description="List of sub-agent calls to make"
    )
    reasoning: str = Field(description="Short explanation of the plan")
    next_node: str = Field(
        description="The next node to transition to. Use 'tool_executor' if agent calls are needed, otherwise 'synthesizer'"
    )


def build_system_prompt(agent_cards: dict) -> str:
    """Build system prompt using agent card descriptions instead of hardcoded text."""
    # Extract agent capabilities from agent cards
    agents_context = ""
    if agent_cards:
        for agent_name, card in agent_cards.items():
            name = card.get("name", agent_name)
            description = card.get("description", "")
            agents_context += f"- {name}: {description}\n\n"
    else:
        # Fallback if no agent cards available
        agents_context = (
            "- Technical Analysis Agent: For price action, indicators (RSI, SMA, MACD), and charts.\n"
            "- Fundamental Analysis Agent: For earnings, balance sheets, ratios, and valuation.\n"
            "- News Analysis Agent: For recent headlines, sentiment analysis, and market events.\n"
        )

    return (
        "You are a Financial Planning Agent. "
        "Your job is to analyze the user's query and the current state to decide which specialized sub-agents need to be consulted.\n\n"
        "AVAILABLE AGENTS:\n"
        f"{agents_context}\n"
        "GUIDELINES:\n"
        "- You can call up to 3 agents in one step.\n"
        "- Analyze any previous 'ToolMessages' to see what data has already been fetched.\n"
        "- If 'feedback' from the synthesizer is present in the state, only address the gaps it highlights.\n"
        "- If no more agent calls are needed to fulfill the user query, set next_node to 'synthesizer'.\n"
        "- If agent calls are required, set next_node to 'tool_executor'."
    )


async def planner(state: State, model: Runnable) -> Command:
    "Planner node: Decides whether to call sub-agents or proceed to synthesis."

    messages = state.get("messages", [])
    agent_cards = state.get("agent_cards", {})

    # Build system prompt from actual agent card descriptions
    system_prompt = build_system_prompt(agent_cards)
    logger.info(
        f"[Advisor Agent Planner] System prompt constructed with agent cards: {system_prompt}"
    )

    # Extract relevant context from state
    user_query = ""
    # first human message in this session or the latest one
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            user_query = m.content
            break
    prompt_content = f"User Query: {user_query}\n"

    # Add summary of ToolMessages
    tool_results = [m.content for m in messages if isinstance(m, ToolMessage)]
    if tool_results:
        prompt_content += "Current findings from previous tool calls:\n" + "\n".join(
            tool_results
        )
    # synthesizer feedback if any
    feedback = state.get("synthesizer_feedback", "")
    if feedback:
        prompt_content += f"Synthesizer Feedback/Gaps: {feedback}\n"

    try:
        # Initialize the model with structured output
        planner_llm = model.with_structured_output(
            PlannerOutput, method="json_schema", strict=True
        )
        response = await planner_llm.ainvoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt_content),
            ]
        )
        logger.info(
            f"[Advisor Agent Planner] Raw response: {response}, Type: {type(response)}"
        )
        # Check if response is valid
        if response is None:
            logger.error("[Advisor Agent Planner] Model returned None response")
            return Command(
                goto="synthesizer",
                update={
                    "messages": [
                        AIMessage(content="Error: No valid response from planner model")
                    ]
                },
            )

        # Prepare tool calls if any
        tool_calls = []
        if response.agent_calls:
            for call in response.agent_calls:
                tool_calls.append(
                    ToolCall(
                        name=call.agent_name,
                        args={"query": call.agent_query},
                        id=f"call_{uuid.uuid4().hex[:8]}",
                    )
                )
        logger.info(
            f"[Advisor Agent Planner] Decided on next node: {response.next_node} with reasoning: {response.reasoning}"
        )
        return Command(
            goto=response.next_node,
            update={
                "messages": [
                    AIMessage(
                        content=f"Plan: {response.reasoning}", tool_calls=tool_calls
                    )
                ]
            },
        )
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

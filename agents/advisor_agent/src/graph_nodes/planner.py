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


PLANNER_SYSTEM_PROMPT = (
    "You are a Financial Planning Agent. "
    "Your job is to analyze the user's query and the current state to decide which specialized sub-agents (Technical, Fundamental, News) need to be consulted."
    "AVAILABLE AGENTS:"
    "1. Technical Analysis Agent: For price action, indicators (RSI, SMA, MACD), and charts."
    "2. Fundamental Analysis Agent: For earnings, balance sheets, ratios, and valuation."
    "3. News Analysis Agent: For recent headlines, sentiment analysis, and market events."
    "GUIDELINES:"
    "- You can call up to 3 agents in one step."
    "- Analyze any previous 'ToolMessages' to see what data has already been fetched."
    "- If 'feedback' from the synthesizer is present in the state, only address the gaps it highlights."
    "- If no more agent calls are needed to fulfill the user query, set next_node to 'synthesizer'."
    "- If agent calls are required, set next_node to 'tool_executor'."
)


async def planner(state: State, model: Runnable) -> Command:
    "Planner node: Decides whether to call sub-agents or proceed to synthesis."

    messages = state.get("messages", [])

    # Extract relevant context from state
    user_query = ""
    # Usually the first human message in this session or the latest one
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            user_query = m.content
            break

    # synthesizer feedback might be stored in a specific key or as an instructions message
    # Assuming it's in state["synthesizer_feedback"] for this example
    feedback = state.get("synthesizer_feedback", "")

    # Construct prompt
    prompt_content = f"User Query: {user_query}\n"
    if feedback:
        prompt_content += f"Synthesizer Feedback/Gaps: {feedback}\n"

    # Add summary of what we already have from ToolMessages
    tool_results = [m.content for m in messages if isinstance(m, ToolMessage)]
    if tool_results:
        prompt_content += "Current findings from previous tool calls:\n" + "\n".join(
            tool_results
        )
    try:
        # Initialize the model with structured output
        planner_llm = model.with_structured_output(PlannerOutput)
        response: PlannerOutput = await planner_llm.ainvoke(
            [
                SystemMessage(content=PLANNER_SYSTEM_PROMPT),
                HumanMessage(content=prompt_content),
            ]
        )

        # Debug logging
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
        # Return Command to control flow
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

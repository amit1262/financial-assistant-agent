"""Generator node that produces responses using conversation context and message history"""

from langgraph.graph import END
from pydantic import BaseModel, Field
from typing import Optional
from langgraph.types import Command
from src.state import State
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import Runnable
import logging

logger = logging.getLogger(__name__)


class SynthesizerOutput(BaseModel):
    response: Optional[str] = Field(
        description="The original response to the user if all required information is present"
    )
    key_points: Optional[str] = Field(description="Key points summarizing the response")
    is_complete: bool = Field(
        description="True if the user query can be fully answered with current data"
    )
    feedback: Optional[str] = Field(
        description="Description of what is missing or needs clarification if not complete"
    )


SYNTHESIZER_SYSTEM_PROMPT = (
    "You are a Financial Synthesizer Agent. Your job is to analyze the user's query and the data fetched from specialized agents to provide a final response.\n\n"
    "CRITICAL INSTRUCTIONS:\n"
    "1. Analysis: Look at the current ToolMessages and user query. Can you answer the query COMPLETELY and ACCURATELY?\n"
    "2. Completeness Check: \n"
    "   - If information is missing (e.g., user asked for fundamental data but you only have technicals), set is_complete=False.\n"
    "   - If the data is present but needs clarification to be useful, set is_complete=False.\n"
    "   - If query is fully answerable, set is_complete=True.\n"
    "3. Feedback: If is_complete=False, specify exactly what is missing or what needs to be fetched in the 'feedback' field.\n"
    "4. Response: If is_complete=True, provide a concise, factual response and key points. If False, you can leave these empty or provide a partial update."
)


async def synthesizer(state: State, model: Runnable) -> Command:
    """Analyze info completeness and either generate final response or provide feedback to planner."""
    try:
        messages = state.get("messages", [])
        # Initialize LLM with structured output
        structured_llm = model.with_structured_output(SynthesizerOutput)
        # Build context from messages
        user_query = ""
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                user_query = m.content
                break

        # Collect tool data with agent attribution
        tool_data = []
        for m in messages:
            if isinstance(m, ToolMessage):
                agent_name = m.name or "unknown_agent"
                tool_data.append(
                    f"Agent name: [{agent_name}] analysis agent. Market data: \n{m.content}"
                )

        prompt_content = f"User Query: {user_query}\n\nMarket Data (from agents):\n"
        prompt_content += (
            "\n---\n".join(tool_data) if tool_data else "No market data available yet."
        )
        logger.info(
            f"[Advisor Agent Synthesizer] Checking completeness for query: {user_query[:50]}..."
        )
        response: SynthesizerOutput = await structured_llm.ainvoke(
            [
                SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT),
                HumanMessage(content=prompt_content),
            ]
        )
        if response.is_complete:
            logger.info(
                "[Advisor Agent Synthesizer] Determined info is complete. Generating final response."
            )
            # Successfully answered - clear any previous feedback and reset iteration count
            return Command(
                goto=END,
                update={
                    "final_response": response.response,
                    "messages": [AIMessage(content=response.key_points or "")],
                    "synthesizer_feedback": "",  # Clear feedback on successful completion
                    "iteration_count": 0,  # Reset for next query
                },
            )
        else:
            logger.info(
                f"[Advisor Agent Synthesizer] Determined info is incomplete. Feedback: {response.feedback}"
            )
            # Check iteration limit before looping back
            iteration_count = state.get("iteration_count", 0)
            max_iterations = 10

            # Stop if we've already done max_iterations and are about to exceed
            if iteration_count >= max_iterations - 1:
                logger.warning(
                    f"[Advisor Agent Synthesizer] Hit max iteration limit ({max_iterations}). Stopping refinement loop."
                )
                # Force completion with partial response
                return Command(
                    goto=END,
                    update={
                        "final_response": f"Response generation stopped after {max_iterations} refinement attempts. Latest feedback: {response.feedback}",
                        "messages": [
                            AIMessage(content=f"Max iterations reached. Stopping.")
                        ],
                        "synthesizer_feedback": "",
                        "iteration_count": 0,  # Reset for next query
                    },
                )

            # Needs more info - loop back to planner and increment iteration count
            return Command(
                goto="planner",
                update={
                    "synthesizer_feedback": response.feedback,
                    "iteration_count": iteration_count + 1,
                    "messages": [
                        AIMessage(
                            content=f"Consulting sub-agents again: {response.feedback}"
                        )
                    ],
                },
            )

    except Exception as e:
        logger.error(
            f"[Advisor Agent Synthesizer] Error in synthesizer node: {e}", exc_info=True
        )
        return Command(
            goto=END,
            update={"final_response": f"Error during synthesis: {str(e)}"},
        )

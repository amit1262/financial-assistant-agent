"""Generator node that produces responses using conversation context and message history"""

from langgraph.graph import END
from pydantic import BaseModel, Field
from typing import Optional
from langgraph.types import Command
from src.state import State
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import Runnable
import logging

logger = logging.getLogger(__name__)


class SynthesizerOutput(BaseModel):
    response: Optional[str] = Field(description="The final response to the user query.")
    is_complete: bool = Field(
        description="True if the user query can be fully answered with current information. False if more information is needed from specialized agents."
    )
    feedback: Optional[str] = Field(
        description="Feedback for the planner on what specific information is missing or what needs to be clarified in order to answer the user query. Set only if is_complete is False."
    )


SYNTHESIZER_SYSTEM_PROMPT = (
    "You are a Financial Synthesizer Agent. Your job is to analyze the user's query, the message history, "
    "and the data fetched from specialized agents to provide the best possible final response.\n"
    "CRITICAL CONSTRAINT RULES:\n"
    "- GROUNDING: Make sure your response is accurate, complete, and grounded ONLY in the data available from the specialized agents. Do not make assumptions or use external knowledge.\n"
    "- AGENT UNAVAILABILITY IS A HARD BOUNDARY: Read the latest '[Planner response]' messages carefully. If the Planner explicitly states that certain sub-agents are unavailable, broken, or not present, you must accept that those data channels are permanently CLOSED. Do not ask for data from them.\n"
    "- RETRY ERRORS: If you see error messages or rate limits in 'ToolMessages', do not suggest or request that specific agent/tool again. It is dead for this turn loop.\n\n"
    "EXECUTION INSTRUCTIONS:\n"
    "1. Analysis: Review the message history and the user's query. Determine what data has been successfully fetched vs. what data is missing.\n"
    "2. Feasible Completeness Check:\n"
    "   - Set `is_complete=True` if you have all the information required to answer the query.\n"
    "   - Set `is_complete=True` if some information is missing BUT the Planner or ToolMessages indicate that the necessary agents are unavailable, rate-limited, or failed. (You cannot fetch what is broken; finalize the answer with what you have).\n"
    "   - Set `is_complete=False` ONLY if required data is missing AND the relevant sub-agent is active, available, and has not been tried yet for this specific query step.\n"
    "3. Output Formatting:\n"
    "   - If `is_complete=True`: Generate the final `response`. If data was missing due to unavailable agents, explicitly declare this limitation to the user in your response (e.g., 'Note: Technical analysis systems are currently offline, so this report focuses on fundamental data...'). Set `feedback` to null.\n"
    "   - If `is_complete=False`: Set `response` to null, and specify exactly what missing information the planner needs to fetch next in the `feedback` field. Be specific."
)


async def synthesizer(state: State, model: Runnable) -> Command:
    """Analyze info completeness and either generate final response or provide feedback to planner."""
    max_iterations = 10  # Max loops between planner & synthesizer
    try:
        messages = state.get("messages", [])

        # Initialize LLM with structured output
        structured_llm = model.with_structured_output(
            SynthesizerOutput, method="json_schema", strict=True
        )
        prompt = [SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT)]
        prompt.extend(messages)

        # Use ainvoke for structured output parsing - wait for completion
        logger.info("[Synthesizer] Invoking with structured output")
        response = await structured_llm.ainvoke(prompt)

        if response.is_complete:
            logger.info(
                "[Synthesizer] Market data is sufficient. Generating final response."
            )
            # Successfully answered - clear any previous feedback and reset iteration count
            return {
                "final_response": response.response,
                "messages": [
                    AIMessage(content=f"[Synthesizer response]: {response.response}")
                ],
                "iteration_count": 0,  # Reset for next query
                "max_iteration_hit": False,
                "response_complete": True,
            }
        else:
            logger.info(
                f"[Synthesizer] Market data is not sufficient. Feedback: {response.feedback[:100]}"
            )
            # Check iteration limit before looping back
            iteration_count = state.get("iteration_count", 0)
            # Stop if we've already done max_iterations
            if iteration_count >= max_iterations - 1:
                logger.warning(
                    f"[Synthesizer] Hit max iteration limit ({max_iterations}). Stopping refinement loop."
                )
                # Force completion with partial response
                return {
                    "final_response": f"Response generation stopped after {max_iterations} iterations. Last response: {response.response}",
                    "messages": [
                        AIMessage(
                            content=f"[Synthesizer response]: Max iterations reached. Stopping. Last response: {response.response}"
                        )
                    ],
                    "iteration_count": 0,  # Reset for next query
                    "max_iteration_hit": True,
                    "response_complete": False,
                }
            else:
                # Needs more info - loop back to planner and increment iteration count
                return {
                    "iteration_count": iteration_count + 1,
                    "messages": [
                        AIMessage(
                            content=f"Synthesizer needs more info. Feedback: {response.feedback}"
                        )
                    ],
                    "max_iteration_hit": False,
                    "response_complete": False,
                }

    except Exception as e:
        logger.error(f"[Synthesizer] Error in synthesizer node: {e}", exc_info=True)
        return Command(
            goto=END,
            update={"final_response": f"Error during synthesis: {str(e)}"},
        )


# Condition function for synthesizer node to determine next step based on response completeness and iteration count.
def synthesizer_condition(state: State) -> str:
    "Condition function for synthesizer node to determine next step based on response completeness and iteration count."
    if state.get("response_complete"):
        return "end"  # Final response is complete, end the workflow
    elif state.get("max_iteration_hit"):
        return "end"  # Max iteration limit hit, end the workflow
    else:
        return "planner"  # Need more info, loop back to planner

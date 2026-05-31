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
    "You are a Financial Synthesizer Agent. Your job is to analyze the user's query and the data fetched from specialized agents to provide a final response to the user query."
    "Make sure your response is accurate, complete, and grounded only in the data available from the specialized agents."
    "Do not make assumptions or use external knowledge. If the data is insufficient to answer the query, provide specific feedback on what information is missing or what needs to be clarified."
    "CRITICAL INSTRUCTIONS:\n"
    "1. Analysis: Look at the messages history and most recent user query. Can you answer the query COMPLETELY and ACCURATELY?. If yes, go ahead and generate the response."
    "2. Completeness Check: \n"
    "   - If information is missing (e.g., user asked for fundamental data but you only have technicals), set is_complete=False.\n"
    "   - If the data is present but needs clarification to be useful, set is_complete=False.\n"
    "   - If query is fully answerable, set is_complete=True.\n"
    "3. Feedback: If is_complete=False, specify exactly what information is missing or what needs to be clarified. This information would be used by the planner to fetch additional data or refine the query.\n"
    "4. Retry Errors: If you encounter any errors in ToolMessages, don't keep suggesting the same tool. For example, an API rate limit error means that tool is currently unavailable, so no point suggesting it again in next planner iteration. Instead, focus on other tools or clarify the query to work with available data.\n"
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
        response = await structured_llm.ainvoke(prompt)
        if response.is_complete:
            logger.info(
                "[Synthesizer] Market data is sufficient. Generating final response."
            )
            # Successfully answered - clear any previous feedback and reset iteration count
            return {
                "final_response": response.response,
                "messages": [AIMessage(content=response.response)],
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
                            content=f"Max iterations reached. Stopping. Last response: {response.response}"
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

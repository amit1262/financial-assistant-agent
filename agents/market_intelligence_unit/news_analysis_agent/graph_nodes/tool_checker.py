# this graph node check if MCP tools are available with the agent or not. If not, it can trigger a fallback mechanism or end the workflow gracefully.

from langgraph.types import Command
from news_analysis_agent.state import State
import logging
from langchain_core.messages import SystemMessage
from langgraph.graph import END

logger = logging.getLogger(__name__)


async def tool_checker(state: State, mcp_tools) -> Command:
    "Graph node to check mcp tool availability and update state accordingly"

    if mcp_tools is None:
        # No tools available, end workflow or trigger fallback
        logger.warning(
            "[News Agent] No MCP tools available for the agent, ending workflow."
        )
        error_message = "[News Agent] No MCP tools configured/available for the agent."
        # global variables
        fallback_message = "[News Agent] Failed (No MCP Tools Configured/Available)"

        update_content = {
            "messages": SystemMessage(content=fallback_message),
            "error": error_message,
            "status": "failure",
        }
        return Command(update=update_content, goto=END)
    else:
        logger.info(
            f"[News Agent] MCP tools available: {len(mcp_tools)}. Continuing workflow."
        )
        update_content = {
            "status": "success",
            "error": "",
        }
        return Command(update=update_content, goto="generator")

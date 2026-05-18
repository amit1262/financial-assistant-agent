"""Query service - Business logic for agent interactions"""

from langchain_core.messages import HumanMessage
import logging
from services.agent_manager import manager

logger = logging.getLogger(__name__)


async def process_query(user_id: str, user_query: str) -> dict:
    "Process the user query through the agent workflow and return the result with status signal"

    logger.info(f"Processing query: {user_query[:50]}... for user_id: {user_id}")
    try:
        config = {"configurable": {"thread_id": user_id}}
        agent = manager.get_agent("advisor")
        result = await agent.graph.ainvoke(
            {"messages": [HumanMessage(content=user_query)]}, config=config
        )
        answer = result.get("final_response", "")
        # Signal: Empty response from agent
        if not answer:
            logger.error(
                f"Agent returned empty response for user_id: {user_id} and query: {user_query}"
            )
            return {
                "answer": None,
                "status": "empty_response",
                "error_message": "Agent returned empty response",
            }

        logger.info(f"Successfully processed query for user_id: {user_id}")
        return {"answer": answer, "status": "success", "error_message": None}

    except Exception as e:
        logger.error(f"Unexpected error processing query: {e}", exc_info=True)
        return {"answer": None, "status": "error", "error_message": str(e)}

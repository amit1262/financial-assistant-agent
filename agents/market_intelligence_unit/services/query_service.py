"""Query service - Business logic for agent interactions"""

import logging

# from technical_analysis_agent.agent import agent as technical_agent
from services.agent_manager import manager
from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)


async def process_query(agent_name: str, agent_query: str) -> dict:
    "Process the agent query through the corresponding agent workflow and return the result with status signal"
    try:
        logger.info(f"Processing Query. Agent: {agent_name}, Query: {agent_query}")
        if agent_name == "technical":
            # invoke the technical analysis agent workflow
            agent = manager.get_agent(agent_name)
            config = {
                "configurable": {"thread_id": "default_thread"}
            }  # TODO - use real user_id
            agent_msg = {"messages": [AIMessage(content=agent_query)]}
            result = await agent.ainvoke(agent_msg, config=config)
            logger.info(
                f"[Technical Agent] Processed agent query - {agent_query} \nResult: {result}"
            )
            # Only return the content of the last message (the answer)
            # result["messages"] is a list of LangChain messages
            final_answer = (
                result["messages"][-1].content
                if result.get("messages")
                else "No response generated."
            )
            logger.info(
                f"[Technical Agent] Final answer for query - {agent_query} \nAnswer:\n {final_answer}"
            )

            return {"answer": final_answer, "status": "success", "error_message": None}

        elif agent_name == "fundamentals":
            # invoke the fundamentals analysis agent workflow
            pass

        elif agent_name == "news":
            # invoke the sentiment/news analysis agent workflow
            pass
        else:
            logger.warning(f"[QueryService] Unknown agent name: {agent_name}")
            return {
                "answer": None,
                "status": "error",
                "error_message": "Unknown agent name.",
            }

    except Exception as e:
        logger.error(
            f"[QueryService] Error! processing query. Agent Name: {agent_name}, Error: {e}",
            exc_info=True,
        )
        return {"answer": None, "status": "error", "error_message": str(e)}

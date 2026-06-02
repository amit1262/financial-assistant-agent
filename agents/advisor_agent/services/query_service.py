"""Query service - Business logic for agent interactions"""

import json
import asyncio
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from pydantic import BaseModel
import logging
from services.agent_manager import manager

logger = logging.getLogger(__name__)


class LangChainEncoder(json.JSONEncoder):
    """Custom JSON encoder for LangChain message objects, Pydantic models, and other non-serializable types"""

    def default(self, obj):
        # Handle LangChain message objects
        if isinstance(obj, BaseMessage):
            # Build message dict with all fields
            msg_dict = {
                "type": obj.type,
                "content": obj.content,
            }
            # Add optional fields only if they exist on this message type
            if hasattr(obj, "name") and obj.name:
                msg_dict["name"] = obj.name
            if hasattr(obj, "tool_call_id") and obj.tool_call_id:
                msg_dict["tool_call_id"] = obj.tool_call_id

            # Include tool_calls (standard LangChain attribute for AIMessage)
            if hasattr(obj, "tool_calls") and obj.tool_calls:
                msg_dict["tool_calls"] = obj.tool_calls

            # Include agent_calls (custom attribute for sub-agent orchestration)
            if hasattr(obj, "agent_calls") and obj.agent_calls:
                # Convert Pydantic models to dicts
                if isinstance(obj.agent_calls, list):
                    msg_dict["agent_calls"] = [
                        call.dict() if hasattr(call, "dict") else call
                        for call in obj.agent_calls
                    ]
                else:
                    msg_dict["agent_calls"] = obj.agent_calls

            return msg_dict
        # Handle other non-serializable objects
        return str(obj)


async def process_query_stream(user_id: str, user_query: str):
    try:
        config = {"configurable": {"thread_id": user_id}}
        agent = manager.get_agent("advisor")

        # Reset only specific state fields for new query, preserving message history
        initial_state = {
            "messages": [HumanMessage(content=user_query)],
            "final_response": "",
            "response_complete": False,
            "iteration_count": 0,
            "max_iteration_hit": False,
        }

        # Use astream_events to track full state snapshots
        logger.info(f"Starting event stream for user {user_id}")
        async for event in await agent.graph.astream_events(
            input=initial_state, config=config, version="v3"
        ):
            method = event.get("method")
            params = event.get("params", {})
            event_data = params.get("data")
            # Capture full state snapshots via values channel
            if method == "values":
                if isinstance(event_data, dict):
                    # Copy the whole state first
                    state_for_display = event_data.copy()

                    # If agent_cards exists, truncate it for display
                    if "agent_cards" in state_for_display:
                        agent_cards_truncated = {}
                        for key, value in state_for_display["agent_cards"].items():
                            value_str = str(value)
                            if len(value_str) > 100:
                                agent_cards_truncated[key] = value_str[:100] + "..."
                            else:
                                agent_cards_truncated[key] = value
                        # Overwrite with truncated version
                        state_for_display["agent_cards"] = agent_cards_truncated

                    yield json.dumps(
                        {
                            "type": "state_update",
                            "state": state_for_display,
                        },
                        cls=LangChainEncoder,
                    ) + "\n"
        logger.info(f"Event stream completed for user {user_id}")
        # Send completion event
        yield json.dumps(
            {
                "type": "complete",
                "status": "success",
            },
            cls=LangChainEncoder,
        ) + "\n"

    except Exception as e:
        logger.error(f"Unexpected error processing query: {e}", exc_info=True)
        yield json.dumps(
            {
                "type": "complete",
                "status": "error",
                "message": str(e),
            },
            cls=LangChainEncoder,
        ) + "\n"


async def process_query(user_id: str, user_query: str) -> dict:
    "Process the user query through the agent workflow and return the result with status signal"

    logger.info(f"Processing query: {user_query[:50]}... for user_id: {user_id}")
    try:
        config = {"configurable": {"thread_id": user_id}}
        agent = manager.get_agent("advisor")

        # Reset only specific state fields for new query, preserving message history
        initial_state = {
            "messages": [HumanMessage(content=user_query)],
            "final_response": "",
            "response_complete": False,
            "iteration_count": 0,
            "max_iteration_hit": False,
        }
        logger.info(f"Invoking agent for user {user_id}")
        result = await agent.graph.ainvoke(initial_state, config=config)
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

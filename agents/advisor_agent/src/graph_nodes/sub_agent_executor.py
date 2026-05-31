# a node to execute tool calls (sub-agents) and return results

import uuid
import asyncio
import httpx
import grpc
import logging
from langchain_core.messages import AIMessage, ToolMessage
from src.state import State

# A2A client imports
from a2a.client import ClientConfig, create_client, card_resolver
from a2a.types import Message, Part, Role, SendMessageRequest
from a2a.helpers import get_message_text

logger = logging.getLogger(__name__)


async def sub_agent_executor(state: State):
    """
    Executes sub-agent tool calls defined in the latest AIMessage.
    Uses A2A client to communicate with specialized agents.
    """
    messages = state.get("messages", [])
    agent_cards = state.get("agent_cards", {})
    timeout = 120  # seconds

    if not messages:
        return {}

    # Simplify: directly check the last message
    last_message = messages[-1]
    if not isinstance(last_message, AIMessage) or not last_message.agent_calls:
        logger.warning("[Tool Node] No agent calls found in the latest AI message")
        return {}

    tool_outputs = []

    # Use a single httpx client for 2-minute timeout as per test_agent.py
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as httpx_client:
        tasks = []
        for call in last_message.agent_calls:
            agent_name = call.agent_name
            query = call.agent_query
            call_id = call.call_id
            # Get agent card from state
            agent_card_data = agent_cards.get(agent_name)
            if not agent_card_data:
                logger.error(f"[Tool Node] No agent card found for: {agent_name}")
                tool_outputs.append(
                    ToolMessage(
                        content=f"Error: Agent '{agent_name}' is not registered.",
                        name=agent_name,
                        tool_call_id=call_id,
                    )
                )
                continue
            # Create an async task for each agent call
            tasks.append(
                call_sub_agent(
                    agent_name, query, call_id, agent_card_data, httpx_client
                )
            )

        # Execute all calls in parallel
        if tasks:
            results = await asyncio.gather(*tasks)
            tool_outputs.extend(results)

    return {"messages": tool_outputs}


async def call_sub_agent(
    agent_name: str,
    query: str,
    call_id: str,
    agent_card: dict,
    httpx_client: httpx.AsyncClient,
) -> ToolMessage:
    """
    Helper to invoke a single sub-agent via A2A protocol.
    """
    try:
        logger.info(
            f"[Tool Node] Invoking sub-agent '{agent_name}' with query: {query}"
        )
        # Parse agent card dict back to AgentCard object using A2A parser
        agent_card = card_resolver.parse_agent_card(agent_card)
        # Setup A2A Client from config
        config = ClientConfig(
            grpc_channel_factory=grpc.aio.insecure_channel, httpx_client=httpx_client
        )
        client = await create_client(agent_card, client_config=config)
        logger.info(f"[Tool Node] client: {client}. Agent: {agent_name}")

        message = Message(
            role=Role.ROLE_USER,
            message_id=str(uuid.uuid4()),
            parts=[Part(text=query)],
            context_id=str(uuid.uuid4()),
        )
        # Stream response
        full_response_text = ""
        stream = client.send_message(SendMessageRequest(message=message))

        async for event in stream:
            # All data comes through status_update messages
            if event.HasField("status_update"):
                if event.status_update.status.HasField("message"):
                    msg_text = get_message_text(event.status_update.status.message)
                    if msg_text:
                        full_response_text += msg_text + "\n"

            # Task events mark task start, ignore them
            elif event.HasField("task"):
                logger.debug(f"[{agent_name}] Task started: {event.task.id}")

        await client.close()

        full_response_text = full_response_text.strip()
        if not full_response_text:
            full_response_text = f"[No data received from {agent_name}]"
        return ToolMessage(
            content=full_response_text, name=agent_name, tool_call_id=call_id
        )

    except Exception as e:
        logger.error(
            f"[Tool Node] Failed to call agent {agent_name}: {str(e)}",
            exc_info=True,
        )
        return ToolMessage(
            content=f"Error communicating with {agent_name}: {str(e)}",
            name=agent_name,
            tool_call_id=call_id,
        )

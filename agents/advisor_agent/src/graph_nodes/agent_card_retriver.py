# fetch agent cards for the specialist sub-agents (technical, fundamental, news)
import os
import httpx
from google.protobuf.json_format import MessageToDict

from a2a.client import A2ACardResolver
import logging
from src.state import State

logger = logging.getLogger(__name__)


async def cards_retriever(state: State):
    # Load agent URLs from environment variables
    agent_urls = {
        "technical": os.getenv("TECHNICAL_AGENT_URL", ""),
        "fundamental": os.getenv("FUNDAMENTAL_AGENT_URL", ""),
        "news": os.getenv("NEWS_AGENT_URL", ""),
    }

    logger.info(f"[Advisor Agent] agent urls loaded: {agent_urls}")
    agent_cards = {}

    try:
        for agent_name, url in agent_urls.items():
            if not url:
                logger.error(f"[Advisor Agent] URL for {agent_name} agent is not set")
                continue
            url = url.rstrip("/")  # Remove trailing slash
            async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as httpx_client:
                resolver = A2ACardResolver(httpx_client, url)
                card = await resolver.get_agent_card()

                # Convert Protocol Buffer AgentCard to dict for serialization
                agent_cards[agent_name] = MessageToDict(card)
                logger.info(
                    f"[Advisor Agent] Agent: {agent_name}, card resolved from: {url}"
                )

        return {"agent_cards": agent_cards}

    except Exception as e:
        logger.error(f"[Advisor Agent] Error fetching agent cards: {e}", exc_info=True)
        raise e

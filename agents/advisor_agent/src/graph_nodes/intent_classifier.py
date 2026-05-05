"""Intent classifier node that routes queries based on whether RAG is needed"""

"To Do - "
"1. Routing when topic is switched by the user. "
"2. Need more clarification from the user on ambiguous queries."

import json
import logging
from typing import Literal
from src.state import State
from src.model import language_model
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import Command

logger = logging.getLogger(__name__)

# Define intents based on available nodes
INTENTS = {
    "needs_rag": "retriever",  # Route to RAG node for context lookup
    "no_rag": "generator",  # Route directly to LLM
}
# fallback option in case of classification failure
FALLBACK_INTENT = "needs_rag"

CLASSIFICATION_PROMPT = (
    "Analyze the following user query and decide if it needs RAG (database retrieval) or can be "
    "answered directly by the LLM.\n\n"
    "Classification examples:\n"
    "1. Query: Explain P/E ratio and its significance in stock analysis.\n"
    "   Intent: no_rag\n"
    "2. Query: Explain P/E ratio of <ticker> and how it compares to industry peers.\n"
    "   Intent: needs_rag\n"
    "3. Query: Is this right time to invest in <ticker> based on recent news and financials?\n"
    "   Intent: needs_rag\n"
    "4. Query: Is this right time to invest in stock market based on recent news and financials?\n"
    "   Intent: no_rag\n"
    "\nUSER_QUERY_START\n"
    "{query}\n"
    "USER_QUERY_END\n\n"
    'Output Format - Respond in JSON format ONLY: {{"intent": "needs_rag"}} or {{"intent": "no_rag"}}. '
    "No explanations, no extra text, just the JSON with intent key."
)

SYSTEM_PROMPT = (
    "Your role is to carefully analyze the user query and estimate if properly answering it "
    "requires RAG i.e., retrieving any external, semantically relevant data/context from vector database or if "
    "it can be answered directly by the LLM "
    "based on general knowledge, reasoning, and other tools it has access to. \n"
    "LLM Tools - Note that the LLM has access to certain tools as well e.g., open web search. These tools can help "
    "answer questions that require up-to-date information (news, sentiments etc.) or any specific financial concept. \n"
    "RAG - On the other hand, the database (which the RAG/retriever node retrieves from) "
    "contains specific financial "
    "data (for some companies) like previous earning releases, stock performances, "
    "shareholder meeting details, business plans/vision details and so on. \n"
    "Be conservative - if there's any chance that RAG is needed, classify as 'needs_rag' "
    "to ensure the agent retrieves necessary context before answering."
)


def intent_classifier(state: State) -> Command[Literal["retriever", "generator"]]:
    """Classify if query needs RAG (data retrieval) or can be answered directly.

    Uses XML delimiters to protect against prompt injection attacks.
    Returns Command with state update and routing to next node.

    Args:
        state: Current state containing messages

    Returns:
        Command with intent update and goto routing to next node
    """
    try:
        # Get the last user message
        user_query = state["messages"][-1].content
        logger.info(f"Classifying query: {user_query[:50]}...")

        # Create prompt for intent classification with XML delimiters for injection protection
        user_prompt = CLASSIFICATION_PROMPT.format(query=user_query)

        # Call LLM to classify intent
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        response = language_model.invoke(messages)
        logger.info(f"LLM response for intent classification: {response.content}")

        # Parse JSON response with injection-safe fallback
        try:
            result = json.loads(response.content)  # ChatOllama returns AIMessage
            intent = result.get("intent", "needs_rag").lower()
        except Exception as e:
            logger.error(
                f"Failed to parse JSON response: {response.content}. Defaulting to needs_rag. Error: {e}"
            )
            intent = "needs_rag"  # Safe default

        # Validate intent against whitelist (defense-in-depth)
        if intent not in INTENTS:
            logger.error(f"Invalid intent '{intent}', defaulting to needs_rag")
            intent = "needs_rag"

        next_node = INTENTS[intent]
        logger.info(f"Intent: {intent} -> routing to {next_node}")

        # Return Command with state update and routing
        return Command(update={"intent": intent}, goto=next_node)

    except Exception as e:
        logger.error(f"Error classifying intent: {e}", exc_info=True)
        # use fallback option
        logger.info(
            f"Fallback situation - Intent: {FALLBACK_INTENT} -> routing to {INTENTS[FALLBACK_INTENT]}"
        )
        return Command(
            update={"intent": FALLBACK_INTENT}, goto=INTENTS[FALLBACK_INTENT]
        )

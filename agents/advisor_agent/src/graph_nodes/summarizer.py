"""Summarizer node that condenses conversation history"""

import logging
from src.state import State
from src.model import language_model
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from langgraph.types import Command

logger = logging.getLogger(__name__)

# Configurable thresholds
MESSAGE_THRESHOLD = 10  # Trigger summarization when messages exceed this
MESSAGES_TO_SUMMARIZE = 5  # Number of older messages to summarize

SUMMARIZATION_SYSTEM_PROMPT = (
    "Your role is to carefully analyze the list of messages and create a short summary that is semantically "
    "equivalent to the original messages but much more concise. This summary would be used to keep the context small for long conversations while retaining the key information."
    "Focus on capturing the key points and topics discussed. If an existing summary is provided, "
    "update it to include the new messages while maintaining the overall narrative. Be brief, clear, and stick to the facts in the messages or existing summary."
)


def summarizer(state: State) -> dict:
    """Summarize older messages when conversation gets long.

    If message count >= MESSAGE_THRESHOLD, takes the oldest MESSAGES_TO_SUMMARIZE
    messages, passes them to LLM for summarization, and updates current_topic_summary.

    Args:
        state: Current state containing messages

    Returns:
        dict with updated current_topic_summary if summarization occurred
    """
    try:
        messages = state.get("messages", [])
        existing_summary = state.get("current_topic_summary", "")

        # Only summarize if we have enough messages
        if len(messages) < MESSAGE_THRESHOLD:
            logger.info(
                f"Message count: ({len(messages)}) below threshold: ({MESSAGE_THRESHOLD}), skipping summarization"
            )
            return {}

        logger.info(
            f"Message count: ({len(messages)}) exceeds threshold: ({MESSAGE_THRESHOLD}). Summarizing oldest {MESSAGES_TO_SUMMARIZE} messages."
        )
        # Extract oldest MESSAGES_TO_SUMMARIZE messages
        messages_to_summarize = messages[:MESSAGES_TO_SUMMARIZE]

        # Format messages for summarization
        conversation_text = "\n".join(
            f"{msg.type}: {msg.content}" for msg in messages_to_summarize
        )
        # logger.debug(f"Messages to summarize: {conversation_text[:100]}...")
        # Build prompt with existing summary context if available
        if existing_summary:
            user_prompt = (
                f"Existing summary:\n{existing_summary}\n\n"
                f"New messages to incorporate:\n{conversation_text}\n\n"
                f"Please update the summary to include these new messages."
            )
            logger.info("Updating existing summary with new messages")
        else:
            user_prompt = (
                f"Please summarize the following messages:\n\n{conversation_text}"
            )
            logger.info("Creating new summary from messages")

        # Call LLM to summarize
        llm_messages = [
            SystemMessage(content=SUMMARIZATION_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
        summary = language_model.invoke(llm_messages)
        updated_summary = summary.content
        logger.info(f"Generated summary: {updated_summary[:100]}...")
        # Create RemoveMessage objects for messages to be removed
        messages_to_remove = [RemoveMessage(id=msg.id) for msg in messages_to_summarize]
        logger.info(
            f"Removed {len(messages_to_remove)} messages from state. Remaining: {len(messages) - len(messages_to_remove)}"
        )
        # Return updated state with new summary and RemoveMessage objects
        return {
            "current_topic_summary": updated_summary,
            "messages": messages_to_remove,
        }

    except Exception as e:
        logger.error(f"Error in summarizer node: {e}", exc_info=True)
        return {}

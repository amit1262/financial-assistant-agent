"""Query refinement node that enhances user queries using conversation context"""

import logging
from src.state import State
from src.model import language_model
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage

logger = logging.getLogger(__name__)

# Number of recent messages to use for context
CONTEXT_WINDOW_SIZE = 3

REFINEMENT_SYSTEM_PROMPT = (
    "You are a query refinement assistant. Your task is to enhance and clarify the user's query "
    "by incorporating relevant context from the conversation history. The refined query should be "
    "complete and self-contained, clearly expressing the user's intent without requiring additional context. "
    "Strictly avoid rephrasing the query and also avoid adding any new information that the user has not mentioned. \n"
    "Also look for any spelling mistakes and grammatical errors and correct them in the refined query. \n"
    "Return only the refined query, nothing else."
)


def query_refiner(state: State) -> dict:
    """Refine user query using conversation context.

    Takes the current user query and last few messages to create a comprehensive,
    self-contained query that incorporates relevant context from the conversation.

    Args:
        state: Current state containing messages

    Returns:
        Command with updated messages list (original user message replaced with refined query)
    """
    try:
        messages = state.get("messages", [])
        current_summary = state.get("current_topic_summary", "")

        if not messages:
            logger.warning("No messages in state, skipping query refinement")
            return {}

        # Get the last message
        last_message = messages[-1]

        # Ensure the last message is a HumanMessage (user query)
        if not isinstance(last_message, HumanMessage):
            logger.warning(
                f"Last message is not HumanMessage (type: {type(last_message).__name__}), skipping refinement"
            )
            return {}

        user_query = last_message.content
        logger.info(f"Original query: {user_query[:50]}...")

        # Get context from recent messages (up to CONTEXT_WINDOW_SIZE)
        # Use min(CONTEXT_WINDOW_SIZE, available_messages) to handle cases with few messages
        num_context = min(CONTEXT_WINDOW_SIZE, len(messages) - 1)

        # Build context if available, otherwise use empty context
        full_context = ""
        if num_context > 0:
            context_messages = messages[-num_context - 1 : -1]
            context_text = "\n".join(
                f"{msg.type}: {msg.content}" for msg in context_messages
            )
            logger.info(
                f"Using {len(context_messages)} context messages for refinement"
            )

            # Build context including topic summary if available
            full_context = context_text
            if current_summary and current_summary.strip():
                full_context = f"Earlier conversation summary (both user and assistant responses):\n{current_summary}\n\nRecent conversation:\n{context_text}"
                logger.info(
                    "Including current_topic_summary in query refinement context"
                )
        else:
            logger.info(
                "First message in conversation - refining for grammar and clarity only"
            )

        # Create LLM prompt for refinement
        llm_messages = [
            SystemMessage(content=REFINEMENT_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Conversation Summary:\n{full_context}\n\n"
                    f"Current query to refine: {user_query}\n\n"
                )
            ),
        ]

        # Call LLM to refine query
        refined_response = language_model.invoke(llm_messages)
        refined_query = refined_response.content

        logger.info(f"Refined query: {refined_query}...")

        # Remove old message and add refined message
        remove_old = RemoveMessage(id=last_message.id)
        refined_message = HumanMessage(content=refined_query)

        logger.info(
            "Query refinement complete. Old HumanMessage removed and replaced with refined version."
        )

        return {"messages": [remove_old, refined_message]}

    except Exception as e:
        logger.error(f"Error in query_refiner node: {e}", exc_info=True)
        return {}

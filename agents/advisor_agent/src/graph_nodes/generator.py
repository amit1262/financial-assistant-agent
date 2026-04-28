"""Generator node that produces responses using conversation context and message history"""

import logging
import json
from src.state import State
from src.model import language_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

logger = logging.getLogger(__name__)

GENERATOR_SYSTEM_PROMPT = (
    "Your role is to carefully analyse the user query and generate an accurate and factual"
    "response. You have access to recent conversation, you might use it to understand what has been discussed so far. "
    "your response should be strickly concise, complete, accurate and factually correct. "
    "Strictly follow the following output format and do not deviate from it under any circumstances."
    '\nOutput format: \n{"response": "your original response here", "key-points": "not more than 5 points summarizing the key information in the response."}'
)


def generator(state: State):
    """Generate response using conversation history and context summary.

    Uses all previous messages (excluding current user query) and the current_topic_summary
    to build context, then generates a response to the user's current query.

    Args:
        state: Current state containing messages, current_topic_summary, and retrieved_context

    Returns:
        dict with messages list updated with the generated AIMessage response
    """
    try:
        messages = state.get("messages", [])
        current_summary = state.get("current_topic_summary", "")
        # retrieved_context = state.get("retrieved_context", [])

        if not messages:
            logger.warning("No messages in state, skipping generation")
            return {}

        # Extract last user message (current query)
        last_message = messages[-1]
        user_query = last_message.content
        logger.info(f"Generating response for query: {user_query[:50]}...")

        # Get all previous messages for context (everything except last user message)
        previous_messages = messages[:-1]

        # Build LLM messages list with XML markers for injection protection
        llm_messages = [SystemMessage(content=GENERATOR_SYSTEM_PROMPT)]

        # Add earlier conversation summary marker (with content if available)
        summary_content = ""
        if current_summary and current_summary.strip():
            summary_content = f"\n{current_summary}"
            logger.info("Including current_topic_summary in context")
        llm_messages.append(
            AIMessage(
                content=f"\nEarlier conversation summary (both user and assistant responses):{summary_content}\n"
            )
        )

        # Add previous messages marker
        llm_messages.append(AIMessage(content="\nPrevious conversation :\n"))
        # Add previous messages as actual message objects (preserve conversation structure)
        if previous_messages:
            llm_messages.extend(previous_messages)
            logger.info(f"Added {len(previous_messages)} previous messages to context")
        else:
            logger.info("No previous messages available (likely first query)")

        # Add retrieved context if available (from RAG)
        # if retrieved_context:
        #     retrieved_text = "\n".join(doc.page_content for doc in retrieved_context)
        #     llm_messages.append(HumanMessage(content=f"Retrieved context:\n{retrieved_text}"))
        #     logger.info(f"Added {len(retrieved_context)} retrieved documents to context")

        # Add current user query with XML markers for injection protection
        llm_messages.append(
            HumanMessage(content=f"\nUSER_QUERY_START\n{user_query}\nUSER_QUERY_END")
        )

        # Call LLM to generate response
        llm_response = language_model.invoke(llm_messages)
        response_content = llm_response.content

        # Parse the JSON response to extract response and key-points
        try:
            parsed_response = json.loads(response_content)
            user_response = parsed_response.get("response", "")
            key_points = parsed_response.get("key-points", "")

            logger.info(f"Parsed LLM response - User response: {user_response}...")
            logger.info(f"Key points for history: {key_points}...")

            # Add key-points to message history instead of full response
            key_points_message = AIMessage(content=f"{key_points}")

            return {"messages": [key_points_message], "final_response": user_response}
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON response from LLM: {response_content}")
            # Fallback: return full response if JSON parsing fails
            return {"messages": [llm_response], "final_response": response_content}

    except Exception as e:
        logger.error(f"Error in generator node: {e}", exc_info=True)
        return {}

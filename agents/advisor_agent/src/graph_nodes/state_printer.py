"""State printer node for debugging - pretty prints the entire state"""

import logging
import json
from src.state import State

logger = logging.getLogger(__name__)


async def pretty_print_state(state: State) -> dict:
    """Pretty print the current state for debugging purposes.
    This is a final debugging node that displays all state information
    in a readable format. Does not modify state.
    """
    try:
        logger.info("\n" + "=" * 80)
        logger.info("STATE PRINTER - FINAL STATE")
        logger.info("=" * 80)

        # Print messages
        messages = state.get("messages", [])
        logger.info(f"\nMESSAGES ({len(messages)} total):")
        logger.info("-" * 80)
        for i, msg in enumerate(messages, 1):
            msg_type = type(msg).__name__
            content_preview = (
                msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            )
            logger.info(f"  [{i}] {msg_type}: {content_preview}")

        # Print current topic summary
        current_summary = state.get("current_topic_summary", "")
        logger.info(f"\nCURRENT TOPIC SUMMARY:")
        logger.info("-" * 80)
        if current_summary and current_summary.strip():
            summary_preview = (
                current_summary[:200] + "..."
                if len(current_summary) > 200
                else current_summary
            )
            logger.info(f"  {summary_preview}")
        else:
            logger.info("  (empty)")

        # Print final response
        final_response = state.get("final_response", "")
        logger.info(f"\nFINAL RESPONSE:")
        logger.info("-" * 80)
        if final_response:
            response_preview = (
                final_response[:200] + "..."
                if len(final_response) > 200
                else final_response
            )
            logger.info(f"  {response_preview}")
        else:
            logger.info("  (empty)")

        # Print any other keys in state
        other_keys = set(state.keys()) - {
            "messages",
            "current_topic_summary",
            "final_response",
        }
        if other_keys:
            logger.info(f"\nOTHER STATE KEYS:")
            logger.info("-" * 80)
            for key in sorted(other_keys):
                value = state.get(key)
                if isinstance(value, (dict, list)):
                    value_preview = json.dumps(value, default=str)[:100]
                else:
                    value_preview = str(value)[:100]
                logger.info(f"  {key}: {value_preview}")

        logger.info("=" * 80 + "\n")

        return {}

    except Exception as e:
        logger.error(f"Error in state_printer node: {e}", exc_info=True)
        return {}

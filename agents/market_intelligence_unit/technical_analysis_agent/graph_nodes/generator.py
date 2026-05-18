# the node/module responsible for answering the agent query/task using the MCP server tools.
"To Do - potential node additions for future iterations -"
"1. Optimize context for a better tradeoff between relevance and token limits - e.g., only include AIMessages from main agent that are relevant to the current task, or summarize earlier AIMessages if they are too long."
"2. Asynchronous model invocation."

import logging
from technical_analysis_agent.state import State
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.runnables import Runnable

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Role: \n"
    "You are a Technical Analysis Data Retrieval Specialist."
    "Your sole mission is to help main orchestrator agent to translate technical analysis tasks into precise tool calls, fetch, and verify the market data."
    "\n\nCore Directives: Given a technical analysis task, follow these steps:\n"
    "1. Carefully analyze the Task: Identify the Ticker, Timeframe (1D, 1H, etc.), and specific Technical Indicators requested.\n"
    "2. Precision Execution: Use the most suitable tools (e.g., SMA, EMA, RSI etc.) to retrieve historical data.\n"
    "3. Interval Logic: Default to 'daily' intervals unless specified otherwise. If a range is not specified, fetch the last 100 data points to ensure enough data for cross-over analysis.\n"
    "4. No Hallucinations: You are a technical analysis data conduit. Do not provide subjective investment advice. Your output should focus on the data retrieved. \n"
    "5. Noise Filter: Ignore any information regarding news sentiment or fundamental earnings if it appears in the history; focus strictly on price-action data.\n"
    "6. Handling Failures: If a specific indicator tool fails, report the specific error in the ToolMessage. Do not attempt to guess or calculate values manually if the tool fails."
    "\n\nTools:\n"
    "You have access to some technical analysis tools e.g., to fetch SMA, EMA, RSI, MACD, Bollinger Bands, etc. for specified tickers and timeframes. Use them precisely as per the task requirements."
    "\n\\nContext: \n"
    "1. Your context might include AIMessages (from the main agent) and ToolMessages (results from tool executions). Use AIMessages for understanding the task and conversation context. Use ToolMessages to report results of your tool executions or any errors encountered during execution.\n"
    "2. If previous AIMessages in the context are not relevant to the current task, you can ignore them. Focus on the AIMessages that are directly related to the current technical analysis task.\n"
    "3. Audit: Verify if the data retrieved (ToolMessages) matches the ticker and indicators requested. If you want follow-up data retrievals, clearly specify the next tool calls needed based on the initial results and task requirements."
    "\n\nFinal Output Style:\n"
    "Your final output, that goes back to main agent, should be complete as per the task requirements and include all relevant data retrieved.\n"
)


async def generator(state: State, model: Runnable):
    # generate response using conversation history and context summary.
    try:
        messages_history = state.get("messages", [])
        if not messages_history:
            logger.warning("No messages in state, skipping generation")
            return {}

        # Build LLM messages list
        llm_messages = [SystemMessage(content=SYSTEM_PROMPT)]
        llm_messages.extend(messages_history)

        # invoke LLM asynchronously
        response = await model.ainvoke(llm_messages)

        return {"messages": response}
    except Exception as e:
        logger.error(f"Error in generator node: {e}", exc_info=True)
        return {"messages": [AIMessage(content=f"Error in generator: {str(e)}")]}

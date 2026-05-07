# the node/module responsible for answering the agent query/task using the MCP server tools.
"To Do - potential node additions for future iterations -"
"1. Optimize context for a better tradeoff between relevance and token limits - e.g., only include AIMessages from main agent that are relevant to the current task, or summarize earlier AIMessages if they are too long."
"2. Asynchronous model invocation."

import logging
from news_analysis_agent.state import State
from news_analysis_agent.model import model
from services.mcp_client_manager import mcp_manager
from langchain_core.messages import SystemMessage, AIMessage

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "Role: \n"
    "You are a News and Sentiment Retrieval Specialist. "
    "Your sole mission is to help the main orchestrator agent translate market news and sentiment tasks into precise tool calls, fetch, and verify recent news and social sentiment data.\n"
    "\n\nCore Directives: Given a news analysis task, follow these steps:\n"
    "1. Carefully analyze the Task: Identify the Ticker, the Date Range (e.g., last 24 hours, last week), and specific topics of interest "
    "(e.g., earnings announcements, management changes, macro-economic impact).\n"
    "2. Precision Execution: Use the most suitable tools (e.g., get_news_feed, get_sentiment_scores) to retrieve the latest headlines and metadata.\n"
    "3. Volume Logic: If a specific number of articles is not specified, fetch the top 10-15 most relevant/recent headlines to provide a representative sample of market sentiment.\n"
    "4. No Hallucinations: You are a news conduit. Do not invent news stories, rumors, or sentiment scores. "
    "Your output should focus strictly on the headlines and scores provided by the tools.\n"
    "5. Noise Filter: Ignore any information regarding technical indicators (RSI, SMA) or deep financial statement metrics (Debt-to-Equity) "
    "if it appears in the history; focus strictly on news text and sentiment signals.\n"
    "6. Handling Failures: If a news tool returns no results for a specific ticker, report this clearly. Do not attempt to summarize news from your general training data; only use data from the ToolMessages.\n"
    "\n\nTools:\n"
    "You have access to market news tools e.g., to fetch real-time news feeds, historical news archives, and sentiment analysis scores "
    "for specified tickers and timeframes. Use them precisely as per the task requirements.\n"
    "\n\nContext: \n"
    "1. Your context might include AIMessages (from the main agent) and ToolMessages (results from tool executions). "
    "Use AIMessages for understanding the scope of the news search. Use ToolMessages to report the actual headlines or sentiment data.\n"
    "2. If previous AIMessages are irrelevant (e.g., discussing technical chart patterns), ignore them. Focus on the specific news/sentiment query.\n"
    "3. Audit: Verify if the news retrieved (ToolMessages) is actually about the requested ticker and falls within the requested timeframe. "
    "If the news is too generic, specify follow-up calls for more targeted sources.\n"
    "\n\nFinal Output Style:\n"
    "Your final output, that goes back to the main agent, should be complete as per the task requirements and include a structured summary of headlines, publication dates, and sentiment scores retrieved.\n"
)


async def generator(state: State):
    # generate response using conversation history and context summary.
    try:
        mcp_client = mcp_manager.get_client("news")
        mcp_tools = (
            await mcp_client.get_mcp_tools()
        )  # async call to fetch tools from MCP client
        model_with_tools = model.bind_tools(mcp_tools)

        messages_history = state.get("messages", [])
        if not messages_history:
            logger.warning("[News Agent] No messages in state, skipping generation")
            return {}

        # Build LLM messages list
        llm_messages = [SystemMessage(content=SYSTEM_PROMPT)]
        llm_messages.extend(messages_history)

        # invoke LLM asynchronously
        response = await model_with_tools.ainvoke(llm_messages)

        return {"messages": response}
    except Exception as e:
        logger.error(f"[News Agent] Error in generator node: {e}", exc_info=True)
        return {"messages": [AIMessage(content=f"Error in generator: {str(e)}")]}

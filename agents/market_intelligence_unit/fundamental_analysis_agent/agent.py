"agent state graph definition."

"To Do - potential node additions for future iterations - "
"1. Add thought trace extraction node after generator to capture intermediate reasoning steps"
"2. if same tool calls are made in a loop, add detection and loop breaking mechanism to avoid infinite cycles. Use State."
"3. Add heartbeat to monitor MCP servers."

import logging
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from fundamental_analysis_agent.state import State
from langgraph.prebuilt import ToolNode, tools_condition
from services.mcp_client_manager import mcp_manager
from fundamental_analysis_agent.graph_nodes.generator import generator
from fundamental_analysis_agent.graph_nodes.tool_checker import tool_checker

logger = logging.getLogger(__name__)


# Build LangGraph workflow
async def build_agent_graph():
    # workflow graph definition
    workflow = StateGraph(state_schema=State)

    # Get tools asynchronously for the ToolNode
    mcp_client = mcp_manager.get_client("fundamental")
    mcp_tools = await mcp_client.get_mcp_tools()

    # add graph nodes
    workflow.add_node("tool_checker", tool_checker)
    workflow.add_node("generator", generator)
    workflow.add_node("tool_node", ToolNode(tools=mcp_tools))

    # add edges
    workflow.add_edge(
        START, "tool_checker"
    )  # can either goto END or generator based on tool_checker output
    workflow.add_conditional_edges(
        "generator", tools_condition, {"tools": "tool_node", "__end__": END}
    )
    workflow.add_edge("tool_node", "generator")

    # state persistence setup
    memory = MemorySaver()
    # compile graph
    graph = workflow.compile(checkpointer=memory)

    return graph


def visualize_agent(graph):
    # visualize the graph
    img = graph.get_graph().draw_mermaid_png()
    with open("workflow.png", "wb") as f:
        f.write(img)

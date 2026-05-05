"agent state graph definition."

"To Do - potential node additions for future iterations - "
"1. Check MemorySaver to make it more persistent across runs"
"2. What happens at Scale - multiple users, multiple concurrent runs - how to manage state and memory across runs? (potentially need to add user/session management in state schema and graph nodes)"
"3. Async execution of graph"
"4. Add reasoning models - different for different nodes in graph"

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from src.state import State
from src.graph_nodes.retriever import retriever
from src.graph_nodes.generator import generator
from src.graph_nodes.intent_classifier import intent_classifier
from src.graph_nodes.summarizer import summarizer
from src.graph_nodes.query_refiner import query_refiner
from src.graph_nodes.state_printer import pretty_print_state


# Build LangGraph workflow
def build_agent_graph():
    # workflow graph definition
    workflow = StateGraph(state_schema=State)

    # add graph nodes
    workflow.add_node("query_refiner", query_refiner)
    workflow.add_node("intent_classifier", intent_classifier)
    workflow.add_node("retriever", retriever)
    workflow.add_node("generator", generator)
    workflow.add_node("summarizer", summarizer)
    # workflow.add_node(
    #     "state_printer", pretty_print_state
    # )  # Final node to print state for debugging

    # add edges
    workflow.add_edge(START, "query_refiner")
    workflow.add_edge("query_refiner", "intent_classifier")
    # intent_classifier uses Command (update state and route to next node))
    workflow.add_edge("retriever", "generator")
    workflow.add_edge("generator", "summarizer")
    workflow.add_edge("summarizer", END)
    # workflow.add_edge("summarizer", "state_printer")
    # workflow.add_edge("state_printer", END)

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


agent = build_agent_graph()

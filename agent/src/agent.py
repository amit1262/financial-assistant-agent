from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from src.state import State
from src.graph_nodes.retriever import retriever
from src.graph_nodes.generator import generator
from src.graph_nodes.intent_classifier import intent_classifier


# Build LangGraph workflow
def build_agent_graph():
    # workflow graph definition
    workflow = StateGraph(state_schema=State)

    # add graph nodes
    workflow.add_node("intent_classifier", intent_classifier)
    workflow.add_node("retriever", retriever)
    workflow.add_node("generator", generator)

    # add edges
    workflow.add_edge(START, "intent_classifier")
    # intent_classifier uses Command (update state and route to next node))
    workflow.add_edge("retriever", "generator")
    workflow.add_edge("generator", END)

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

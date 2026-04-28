from src.state import State

# from langchain_core.prompts import SystemMessagePromptTemplate

# from shared.vector_store import vector_store


def retriever(state: State):
    """Get context from the vector index based on the user's query.

    Args:
        state (_type_): current state of the chatbot.

    Returns:
        _type_: context retrieved from the vector index.
    """
    # context = vector_store.similarity_search(query=state["messages"][-1].content, k=1)
    # return {"context": context}
    return {"context": "Placeholder"}

from shared.state import State
from langchain_core.documents import Document


def pretty_print_state(state: State):
    """
    Pretty print the current state of the chatbot.
    This function can be extended to include more complex logic.
    """
    print("\nCurrent State:")
    print("-" * 100)
    for key, value in state.items():
        print(f"{key}:")
        if key == "messages":
            for message in value:
                print(f"{message.type.upper()}: {message.content}\n")
        elif key == "context":
            print(f"Number of documents: {len(value)}")
            for doc in value:
                print(f"Document ID: {doc.metadata.get('id', 'N/A')}")
                print(f"Content: {doc.page_content[:500]}...")
    print("-" * 100)


def pretty_print_context(context: list[Document]):
    """
    Pretty print the context retrieved from the vector index.
    This function can be extended to include more complex logic.
    """
    print("Retrieved Context:")
    print("-" * 40)
    for doc in context:
        print(f"Document ID: {doc.metadata.get('id', 'N/A')}")
        print(f"Content: {doc.page_content[:200]}...")  # Print first 200 characters
        print("-" * 40)

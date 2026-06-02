from typing import Literal, TypedDict, Annotated
from langgraph.graph.message import add_messages


class State(TypedDict):
    "Define the agent state schema"

    messages: Annotated[list, add_messages]
    status: Literal["working", "success", "failure"]
    error: str

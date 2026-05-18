"""API Request and Response models for Fundamental Analysis Agent"""

from os import error
from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for user queries"""

    question: str = Field(..., description="The user's question or prompt")
    user_id: str = Field(
        default="default_user", description="Unique identifier for the user"
    )


class QueryResponse(BaseModel):
    """Response model for agent answers"""

    answer: Any = Field(description="The agent's response")
    status: str = Field(default="success", description="Status of the query processing")
    code: int = Field(default=200, description="HTTP status code for the response")
    error_message: Any = Field(
        default=None, description="Error message if status is not success"
    )

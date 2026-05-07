"""API Request and Response models for Fundamental Analysis Agent"""

from os import error
from typing import Any

from pydantic import BaseModel, Field


class QueryResponse(BaseModel):
    """Response model for agent answers"""

    answer: Any = Field(description="The agent's response")
    status: str = Field(default="success", description="Status of the query processing")
    code: int = Field(default=200, description="HTTP status code for the response")
    error_message: Any = Field(
        default=None, description="Error message if status is not success"
    )

"""API Request and Response models for Financial Assistant Agent"""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for financial queries"""

    user_id: str = Field(
        min_length=1, description="User identifier for conversation persistence"
    )
    question: str = Field(min_length=1, description="The financial question to ask")


class QueryResponse(BaseModel):
    """Response model for agent answers"""

    answer: str = Field(description="The agent's response")
    status: str = Field(default="success", description="Status of the query processing")
    code: int = Field(default=200, description="HTTP status code for the response")
    user_id: str = Field(description="User identifier for conversation persistence")

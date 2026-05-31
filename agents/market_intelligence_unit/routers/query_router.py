from fastapi import APIRouter, HTTPException
import logging
from services import query_service
from services.schemas import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["Market Intelligence Agents"])


# API endpoints
@router.post("/{agent_name}", response_model=QueryResponse)
async def route_agent(agent_name: str, request: QueryRequest) -> QueryResponse:
    """Endpoint to process user queries through a specific market intelligence sub-agent
    Args:
        agent_name: Name of the agent to route the query to (e.g., "technical", "fundamentals", "news")
        request: QueryRequest containing the question/prompt and optional user metadata
    Returns:
        QueryResponse containing the final analysis or diagnostic error details
    """

    try:
        agent_query = request.question
        logger.info(f"Received query for agent {agent_name}: {agent_query}")
        if agent_query.strip() == "":
            logger.warning(f"[QueryRouter] Empty query for agent {agent_name}")
            return {
                "answer": None,
                "status": "error",
                "code": 400,
                "error_message": "Agent query cannot be empty.",
            }
        result = await query_service.process_query(agent_name, agent_query=agent_query)
        # Handle status signal from service
        if result["status"] == "success":
            # Signal: All OK, return 200
            # Return dict for FastAPI to validate against response_model
            return {
                "answer": result["answer"],
                "status": "success",
                "code": 200,
                "error_message": None,
            }

        elif result["status"] == "error":
            # Signal: Exception in agent, return 500
            logger.error(
                f"Agent error for agent {agent_name}: {result['error_message']}"
            )
            raise HTTPException(
                status_code=500,
                detail=f"Agent processing error: {result['error_message']}",
            )

    except Exception as e:
        logger.error(
            f"[QueryRouter] Unexpected error!. Agent Name: {agent_name}, Error: {e}",
            exc_info=True,
        )
        return {"answer": None, "status": "error", "code": 500, "error_message": str(e)}


# "/.well-known/agent.json"
@router.get("/.well-known/agent.json", response_model=list)
async def get_agent_cards():
    "Endpoint to retrieve available agent cards and their metadata"
    return [
        {
            "agent_name": "TechnicalAnalysisAgent",
            "description": "Provides financial advice based on user queries.",
            "capabilities": ["investment advice", "budgeting tips", "market analysis"],
        },
        {
            "agent_name": "StockAnalysisAgent",
            "description": "Analyzes stock market data and provides insights.",
            "capabilities": [
                "stock analysis",
                "market trends",
                "investment recommendations",
            ],
        },
    ]

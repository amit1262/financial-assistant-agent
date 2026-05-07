from fastapi import APIRouter, HTTPException
import logging
from services.schemas import QueryRequest, QueryResponse
from services import query_service

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["Advisor Agent"])


# API endpoints
@router.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest) -> QueryResponse:
    """Endpoint to process user queries through the agent workflow
    Args:
        request: QueryRequest containing the user's question and user_id
    Returns:
        QueryResponse with the agent's answer, status, and HTTP code
    """
    try:
        # Extract user_id for conversation persistence (thread_id)
        user_id = request.user_id
        # Invoke agent - LangGraph automatically manages state
        result = await query_service.process_query(user_id, request.question)

        # Handle status signal from service
        if result["status"] == "success":
            # Signal: All OK, return 200
            return QueryResponse(
                answer=result["answer"], status="success", code=200, user_id=user_id
            )

        elif result["status"] == "empty_response":
            # Signal: Agent returned empty response, return 500
            logger.error(f"Empty response from agent for user {user_id}")
            raise HTTPException(
                status_code=500,
                detail=f"No response generated. Error! : {result['error_message']}",
            )

        elif result["status"] == "error":
            # Signal: Exception in agent, return 500
            logger.error(f"Agent error for user {user_id}: {result['error_message']}")
            raise HTTPException(
                status_code=500,
                detail=f"Agent processing error: {result['error_message']}",
            )

    except Exception as e:
        logger.error(
            f"Unexpected error in query endpoint for user {request.user_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail="Internal server error while processing query"
        )

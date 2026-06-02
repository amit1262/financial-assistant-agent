import logging
from a2a.helpers import (
    new_task_from_user_message,
    new_text_artifact_update_event,
    new_text_status_update_event,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types.a2a_pb2 import (
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from technical_analysis_agent.agent import TechnicalAnalysisAgent
from langchain_core.messages import AIMessage, ToolMessage, SystemMessage

logger = logging.getLogger(__name__)


class TechnicalAnalysisAgentExecutor(AgentExecutor):
    "A2A Executor for Technical Analysis Agent."

    def __init__(self) -> None:
        self.agent = TechnicalAnalysisAgent()

    async def initialize(self):
        "Initialize the agent (load tools, model, graph)."
        await self.agent.initialize()
        return self

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        "Execute the technical analysis agent workflow."

        # 1. Setup Task
        task = context.current_task or new_task_from_user_message(context.message)
        # add task to the queue
        await event_queue.enqueue_event(task)

        # 2. Report Working Status
        await event_queue.enqueue_event(
            new_text_status_update_event(
                task_id=task.id,
                context_id=task.context_id,
                state=TaskState.TASK_STATE_WORKING,
                text="[Technical Executor] Processing task.\n",
            )
        )
        try:
            # 3. Invoke the LangGraph Agent with streaming
            # Extract text from the protobuf Message
            agent_query_text = (
                context.message.parts[0].text
                if context.message.parts
                else str(context.message)
            )
            logger.info(f"[Technical Executor] Received query: {agent_query_text}")

            # Build the input for the graph with text content
            initial_state = {
                "messages": [AIMessage(content=agent_query_text)],
                "status": "working",
                "error": "",
            }
            config = {"configurable": {"thread_id": task.id}}
            # Use astream_events to stream state updates from LangGraph
            final_state = None
            astream_events = await self.agent.graph.astream_events(
                input=initial_state, config=config, version="v3"
            )
            async for event in astream_events:
                method = event.get("method")
                params = event.get("params", {})
                event_data = params.get("data")

                # Capture full state snapshots via values channel
                if method == "values" and isinstance(event_data, dict):
                    final_state = event_data
                    if "messages" in event_data:
                        msg = event_data["messages"][-1]
                        latest_message = ""
                        if isinstance(msg, AIMessage) or isinstance(msg, SystemMessage):
                            if msg.content:
                                latest_message = msg.content[:100]
                            if msg.tool_calls:
                                # 1. LLM is calling tools
                                tools = ", ".join([tc["name"] for tc in msg.tool_calls])
                                latest_message += (
                                    f"\n[Technical Agent] Calling tools: {tools}"
                                )
                        elif isinstance(msg, ToolMessage):
                            # 3. Tool has finished executing
                            latest_message += f"[Technical Executor] Tool '{msg.name}' completed execution."
                        if latest_message:
                            await event_queue.enqueue_event(
                                new_text_status_update_event(
                                    task_id=task.id,
                                    context_id=task.context_id,
                                    state=TaskState.TASK_STATE_WORKING,
                                    text=f"{latest_message}\n",
                                )
                            )
            # 4. Enqueue the Result Artifact - final result is in the last state snapshot
            agent_response = final_state.get("messages", [])[-1].content
            await event_queue.enqueue_event(
                new_text_artifact_update_event(
                    task_id=task.id,
                    context_id=task.context_id,
                    name="technical_analysis_result",
                    text=f"[Technical Executor] Final result: {agent_response}",
                )
            )
            # 5. Report Completion (using v1.0 helper)
            await event_queue.enqueue_event(
                new_text_status_update_event(
                    task_id=task.id,
                    context_id=task.context_id,
                    state=TaskState.TASK_STATE_COMPLETED,
                    text="[Technical Executor] Task Finished.",
                )
            )
        except Exception as e:
            logger.error(f"[Technical Executor] Execution failed: {e}", exc_info=True)
            await event_queue.enqueue_event(
                new_text_status_update_event(
                    task_id=task.id,
                    context_id=task.context_id,
                    state=TaskState.TASK_STATE_FAILED,
                    text=f"[Technical Executor] Technical analysis failed: {e}",
                )
            )
            raise e  # re-raise to ensure upstream handling/logging

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        "Cancel the current execution (not implemented)."
        logger.warning("[Technical Executor] Cancel requested but not supported.")
        raise Exception("cancel not supported")

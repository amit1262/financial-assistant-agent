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
from news_analysis_agent.agent import NewsAnalysisAgent
from langchain_core.messages import AIMessage, ToolMessage

logger = logging.getLogger(__name__)


class NewsAnalysisAgentExecutor(AgentExecutor):
    "A2A Executor for News Analysis Agent."

    def __init__(self) -> None:
        self.agent = NewsAnalysisAgent()

    async def initialize(self):
        "Initialize the agent (load tools, initialize model, build graph)."
        await self.agent.initialize()
        return self

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        "Execute the news analysis agent workflow."

        # 1. Setup Task
        task = context.current_task or new_task_from_user_message(context.message)
        # add task to the queue - protocol requires this
        await event_queue.enqueue_event(task)

        # 2. Report Working Status
        await event_queue.enqueue_event(
            new_text_status_update_event(
                task_id=task.id,
                context_id=task.context_id,
                state=TaskState.TASK_STATE_WORKING,
                text="[News Analysis Executor] Processing task...",
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
            logger.info(
                f"[News Analysis Agent Executor] Received query: {agent_query_text}"
            )

            # Build the input for the graph with text content
            agent_msg = {"messages": [AIMessage(content=agent_query_text)]}
            config = {"configurable": {"thread_id": task.id}}

            # Use astream to stream events from LangGraph
            final_message = ""
            async for event in self.agent.graph.astream(
                agent_msg, config=config, stream_mode="values"
            ):
                if "messages" in event:
                    msg = event["messages"][-1]
                    status_text = ""

                    if isinstance(msg, AIMessage):
                        if msg.tool_calls:
                            # 1. LLM is calling tools
                            tools = ", ".join([tc["name"] for tc in msg.tool_calls])
                            status_text = f"[News Agent] Calling tools: {tools}"
                        elif msg.content:
                            # 2. LLM is providing a final or intermediate answer
                            final_message = msg.content
                            status_text = f"[News Agent] {final_message}..."
                    elif isinstance(msg, ToolMessage):
                        # 3. Tool has finished executing
                        status_text = (
                            f"[News Agent] Tool '{msg.name}' completed execution."
                        )

                    if status_text:
                        await event_queue.enqueue_event(
                            new_text_status_update_event(
                                task_id=task.id,
                                context_id=task.context_id,
                                state=TaskState.TASK_STATE_WORKING,
                                text=status_text,
                            )
                        )

            # 4. Enqueue the Result Artifact
            await event_queue.enqueue_event(
                new_text_artifact_update_event(
                    task_id=task.id,
                    context_id=task.context_id,
                    name="news_analysis_result",
                    text="",
                )
            )
            logger.info(f"[News Agent Executor] Result updated - {final_message}...")
            # 5. Report Completion
            await event_queue.enqueue_event(
                new_text_status_update_event(
                    task_id=task.id,
                    context_id=task.context_id,
                    state=TaskState.TASK_STATE_COMPLETED,
                    text="[News Analysis Executor] News analysis completed successfully.",
                )
            )

        except Exception as e:
            logger.error(
                f"[News Analysis Agent Executor] Execution failed: {e}", exc_info=True
            )
            await event_queue.enqueue_event(
                new_text_status_update_event(
                    task_id=task.id,
                    context_id=task.context_id,
                    state=TaskState.TASK_STATE_FAILED,
                    text=f"[News Analysis Executor] News analysis failed: {e}",
                )
            )
            raise e  # re-raise to ensure upstream handling/logging

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        "Cancel the current execution (not implemented)."
        logger.warning(
            "[News Analysis Agent Executor] Cancel requested but not supported."
        )
        raise Exception("cancel not supported")

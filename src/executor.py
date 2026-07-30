"""A2A Executor: Routes messages to the policy-compliance agent."""
import logging
import uuid

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import InvalidRequestError, TaskState, UnsupportedOperationError
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError

from a2a_compat import await_if_needed, task_context_id
from agent import Agent

logger = logging.getLogger(__name__)

TERMINAL_STATES = {
    TaskState.completed, TaskState.canceled,
    TaskState.failed, TaskState.rejected,
}


class Executor(AgentExecutor):
    def __init__(self):
        self.agent = Agent()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        msg = context.message
        if not msg:
            raise ServerError(error=InvalidRequestError(message="Missing message"))

        task = context.current_task
        if task and task.status.state in TERMINAL_STATES:
            raise ServerError(error=InvalidRequestError(
                message=f"Task {task.id} already processed"))

        if not task:
            task = new_task(msg)
            await await_if_needed(event_queue.enqueue_event(task))

        context_id = task_context_id(task)
        if context_id is None:
            raise ServerError(error=InvalidRequestError(
                message="Task is missing a context identifier"))
        updater = TaskUpdater(event_queue, task.id, context_id)
        await await_if_needed(updater.start_work())

        try:
            await self.agent.run(msg, updater)
            await await_if_needed(updater.complete())
        except Exception:
            error_id = uuid.uuid4().hex[:12]
            logger.exception("Agent task failed error_id=%s", error_id)
            await await_if_needed(
                updater.failed(
                    new_agent_text_message(
                        f"The request failed safely. Reference: {error_id}",
                        context_id=context_id,
                        task_id=task.id,
                    )
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError())

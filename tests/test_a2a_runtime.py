"""Tests against the installed A2A SDK runtime objects."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from a2a.server.events import EventQueue
from a2a.server.agent_execution import RequestContext
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    DataPart,
    Message,
    MessageSendParams,
    Part,
    Role,
    TaskState,
    TextPart,
)
from a2a.utils import new_task


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import executor as executor_module  # noqa: E402
from a2a_compat import (  # noqa: E402
    await_if_needed,
    message_context_id,
    task_context_id,
)


def test_await_if_needed_accepts_synchronous_result() -> None:
    marker = object()

    assert asyncio.run(await_if_needed(marker)) is marker


def test_await_if_needed_accepts_asynchronous_result() -> None:
    async def operation() -> str:
        return "completed"

    assert asyncio.run(await_if_needed(operation())) == "completed"


def test_real_event_queue_enqueue_is_compatible() -> None:
    queue = EventQueue()
    event = object()

    asyncio.run(await_if_needed(queue.enqueue_event(event)))

    assert queue.queue.get_nowait() is event


def _message() -> Message:
    return Message(
        messageId="message-1",
        contextId="context-1",
        role=Role.user,
        parts=[Part(root=TextPart(text="Hello"))],
    )


def test_real_task_context_id_is_read_from_installed_sdk() -> None:
    task = new_task(_message())

    assert task.contextId == "context-1"
    assert task_context_id(task) == "context-1"
    assert message_context_id(_message()) == "context-1"


def test_real_task_updater_lifecycle_is_compatible() -> None:
    queue = EventQueue()
    updater = TaskUpdater(queue, "task-1", "context-1")

    async def exercise() -> None:
        await await_if_needed(updater.start_work())
        await await_if_needed(
            updater.add_artifact(
                parts=[
                    Part(
                        root=DataPart(data={"content": "runtime-compatible"})
                    )
                ],
                name="Response",
            )
        )
        await await_if_needed(updater.complete())

    asyncio.run(exercise())

    events = []
    while not queue.queue.empty():
        events.append(queue.queue.get_nowait())
    assert len(events) == 3
    assert events[0].status.state is TaskState.working
    assert events[1].artifact.name == "Response"
    assert events[2].status.state is TaskState.completed


def test_executor_uses_real_task_context_and_completes(
    monkeypatch,
) -> None:
    class StubAgent:
        async def run(self, message, updater) -> None:
            assert message.contextId == "context-1"
            await await_if_needed(
                updater.add_artifact(
                    parts=[
                        Part(root=DataPart(data={"content": "completed"}))
                    ],
                    name="Response",
                )
            )

    monkeypatch.setattr(executor_module, "Agent", StubAgent)
    message = _message()
    context = RequestContext(
        request=MessageSendParams(message=message),
        context_id="context-1",
    )
    queue = EventQueue()

    asyncio.run(executor_module.Executor().execute(context, queue))

    events = []
    while not queue.queue.empty():
        events.append(queue.queue.get_nowait())
    assert any(
        getattr(event, "artifact", None)
        and event.artifact.name == "Response"
        for event in events
    )
    assert events[-1].status.state is TaskState.completed

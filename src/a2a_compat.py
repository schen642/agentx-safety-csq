"""Compatibility helpers for synchronous and asynchronous A2A SDK releases."""

from __future__ import annotations

import inspect
from typing import Any


async def await_if_needed(result: Any) -> Any:
    """Await an SDK result only when the installed release returns an awaitable."""

    if inspect.isawaitable(result):
        return await result
    return result


def task_context_id(task: Any) -> str | None:
    """Read the task context identifier across A2A SDK naming variants."""

    context_id = getattr(task, "contextId", None)
    if context_id is None:
        context_id = getattr(task, "context_id", None)
    return context_id if isinstance(context_id, str) and context_id else None


def message_context_id(message: Any) -> str | None:
    """Read the message context identifier across A2A SDK naming variants."""

    context_id = getattr(message, "contextId", None)
    if context_id is None:
        context_id = getattr(message, "context_id", None)
    return context_id if isinstance(context_id, str) and context_id else None

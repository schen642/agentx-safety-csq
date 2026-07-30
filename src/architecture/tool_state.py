"""Result-driven tool-call state for architecture Phase C."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ToolCategory(str, Enum):
    LOOKUP = "lookup"
    ACTION = "action"
    DECISION = "decision"


class ToolResultDisposition(str, Enum):
    MATCHED_SUCCESS = "matched_success"
    MATCHED_ERROR = "matched_error"
    UNKNOWN_CALL_ID = "unknown_call_id"
    DUPLICATE_RESULT = "duplicate_result"


@dataclass(frozen=True)
class PendingToolCall:
    call_id: str
    tool_name: str
    arguments: Any
    category: ToolCategory
    request_key: str | None = None


@dataclass(frozen=True)
class CompletedToolCall:
    call: PendingToolCall
    result: Any


@dataclass(frozen=True)
class FailedToolCall:
    call: PendingToolCall
    result: Any


class ToolCallTracker:
    """Reconcile emitted calls with later environment results by call ID."""

    def __init__(self) -> None:
        self.pending: dict[str, PendingToolCall] = {}
        self.completed: dict[str, CompletedToolCall] = {}
        self.failed: dict[str, FailedToolCall] = {}
        self.unknown_result_ids: list[str] = []
        self.duplicate_result_ids: list[str] = []

    def register(self, call: PendingToolCall) -> None:
        """Register an emitted call without treating it as completed."""

        if call.call_id in self.pending:
            raise ValueError(f"duplicate pending tool call ID: {call.call_id}")
        if call.call_id in self.completed or call.call_id in self.failed:
            raise ValueError(f"reused resolved tool call ID: {call.call_id}")
        self.pending[call.call_id] = call

    def resolve(
        self,
        call_id: str,
        result: Any,
        *,
        is_error: bool,
    ) -> tuple[ToolResultDisposition, PendingToolCall | None]:
        """Resolve exactly one call, ignoring unknown and repeated results."""

        if call_id in self.completed or call_id in self.failed:
            self.duplicate_result_ids.append(call_id)
            return ToolResultDisposition.DUPLICATE_RESULT, None

        call = self.pending.pop(call_id, None)
        if call is None:
            self.unknown_result_ids.append(call_id)
            return ToolResultDisposition.UNKNOWN_CALL_ID, None

        if is_error:
            self.failed[call_id] = FailedToolCall(call=call, result=result)
            return ToolResultDisposition.MATCHED_ERROR, call

        self.completed[call_id] = CompletedToolCall(call=call, result=result)
        return ToolResultDisposition.MATCHED_SUCCESS, call

    def completed_ids(
        self,
        category: ToolCategory,
        *,
        request_key: str | None = None,
    ) -> list[str]:
        return [
            call_id
            for call_id, completed in self.completed.items()
            if completed.call.category is category
            and (
                request_key is None
                or completed.call.request_key == request_key
            )
        ]

    def has_completed(
        self,
        category: ToolCategory,
        *,
        request_key: str | None = None,
    ) -> bool:
        return bool(self.completed_ids(category, request_key=request_key))

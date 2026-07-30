"""Deterministic lookup sequencing and retry state for Phase D."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .workflows import (
    KNOWN_LOOKUP_TOOLS,
    available_workflow_steps,
    workflow_for,
)


class InvestigationStatus(str, Enum):
    INVESTIGATING = "investigating"
    COMPLETE = "complete"
    MISSING_EVIDENCE = "missing_evidence"


@dataclass
class InvestigationState:
    request_key: str
    domain: str
    request_type: str
    required_steps: tuple[str, ...]
    steps: tuple[str, ...]
    missing_steps: tuple[str, ...]
    retry_limit: int = 1
    current_index: int = 0
    attempts: dict[str, int] = field(default_factory=dict)
    completed_steps: list[str] = field(default_factory=list)
    failed_steps: list[str] = field(default_factory=list)
    results: dict[str, Any] = field(default_factory=dict)
    pending_call_id: str | None = None
    status: InvestigationStatus = InvestigationStatus.INVESTIGATING
    context_only: bool = False

    @property
    def current_tool(self) -> str | None:
        if self.status is not InvestigationStatus.INVESTIGATING:
            return None
        if self.current_index >= len(self.steps):
            return None
        return self.steps[self.current_index]

    @property
    def missing_evidence(self) -> bool:
        return self.status is InvestigationStatus.MISSING_EVIDENCE

    def register_call(self, call_id: str, tool_name: str) -> None:
        if self.pending_call_id is not None:
            raise ValueError("an investigation call is already pending")
        if tool_name != self.current_tool:
            raise ValueError(
                f"expected investigation tool {self.current_tool!r}, "
                f"got {tool_name!r}"
            )
        self.pending_call_id = call_id
        self.attempts[tool_name] = self.attempts.get(tool_name, 0) + 1

    def accept_success(self, call_id: str, tool_name: str, result: Any) -> None:
        self._validate_result(call_id, tool_name)
        self.pending_call_id = None
        self.completed_steps.append(tool_name)
        self.results[tool_name] = result
        self.current_index += 1
        self._finish_if_exhausted()

    def accept_error(self, call_id: str, tool_name: str) -> None:
        self._validate_result(call_id, tool_name)
        self.pending_call_id = None
        if self.attempts.get(tool_name, 0) > self.retry_limit:
            self.failed_steps.append(tool_name)
            self.status = InvestigationStatus.MISSING_EVIDENCE

    def _validate_result(self, call_id: str, tool_name: str) -> None:
        if call_id != self.pending_call_id:
            raise ValueError(f"unexpected investigation result ID: {call_id}")
        if tool_name != self.current_tool:
            raise ValueError(f"unexpected investigation result tool: {tool_name}")

    def _finish_if_exhausted(self) -> None:
        if self.current_index < len(self.steps):
            return
        self.status = (
            InvestigationStatus.MISSING_EVIDENCE
            if self.missing_steps
            else InvestigationStatus.COMPLETE
        )

    def restart(self) -> None:
        """Re-run available evidence collection after validator rejection."""

        if self.pending_call_id is not None:
            raise ValueError("cannot restart while a lookup is pending")
        self.current_index = 0
        self.completed_steps.clear()
        self.failed_steps.clear()
        self.attempts.clear()
        self.status = (
            InvestigationStatus.INVESTIGATING
            if self.steps
            else (
                InvestigationStatus.COMPLETE
                if self.context_only
                else InvestigationStatus.MISSING_EVIDENCE
            )
        )


def build_investigation(
    *,
    request_key: str,
    domain: str,
    request_type: str,
    available_tool_names: set[str],
    retry_limit: int = 1,
) -> InvestigationState | None:
    """Create state from the registry and the actual runtime tool catalog."""

    required = workflow_for(domain, request_type)
    if not required:
        return None
    steps, missing = available_workflow_steps(
        domain,
        request_type,
        available_tool_names,
    )
    state = InvestigationState(
        request_key=request_key,
        domain=domain,
        request_type=request_type,
        required_steps=required,
        steps=steps,
        missing_steps=missing,
        retry_limit=retry_limit,
    )
    if not steps:
        # Some Pi-Bench scenarios intentionally expose action tools only. In
        # that contract, the supplied task/user context is the complete
        # evidence surface. This is different from a failed or partial lookup
        # workflow, which must remain missing evidence.
        state.context_only = not bool(
            available_tool_names.intersection(KNOWN_LOOKUP_TOOLS)
        )
        state.status = (
            InvestigationStatus.COMPLETE
            if state.context_only
            else InvestigationStatus.MISSING_EVIDENCE
        )
        if state.context_only:
            state.missing_steps = ()
    return state

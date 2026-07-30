"""Deterministic execution queues for architecture Phase F."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .adjudication import ActionSpec, DecisionDraft
from .models import RequestState
from .tool_registry import DOMAIN_TOOL_PRIORITIES, IRREVERSIBLE_TOOLS


class ExecutionStatus(StrEnum):
    READY = "ready"
    WAITING_RESULT = "waiting_result"
    COMPLETE = "complete"
    FAILED_SAFE = "failed_safe"


class StepMutability(StrEnum):
    MUTATING = "mutating"
    IRREVERSIBLE = "irreversible"
    DECISION = "decision"


@dataclass(frozen=True)
class ToolStep:
    tool_name: str
    arguments: dict[str, Any]
    request_key: str
    mutability: StepMutability
    retry_limit: int = 0


@dataclass(frozen=True)
class CompletedToolStep:
    step: ToolStep
    call_id: str
    result: Any


@dataclass
class ExecutionQueue:
    request_key: str
    steps: list[ToolStep]
    current_index: int = 0
    pending_call_id: str | None = None
    attempts: dict[int, int] = field(default_factory=dict)
    completed: list[CompletedToolStep] = field(default_factory=list)
    failed_step: ToolStep | None = None
    status: ExecutionStatus = ExecutionStatus.READY

    @property
    def current_step(self) -> ToolStep | None:
        if self.status is not ExecutionStatus.READY:
            return None
        if self.current_index >= len(self.steps):
            return None
        return self.steps[self.current_index]

    def register_emission(self, call_id: str) -> ToolStep:
        step = self.current_step
        if step is None:
            raise ValueError("no execution step is ready")
        self.pending_call_id = call_id
        self.attempts[self.current_index] = (
            self.attempts.get(self.current_index, 0) + 1
        )
        self.status = ExecutionStatus.WAITING_RESULT
        return step

    def accept_success(self, call_id: str, result: Any) -> CompletedToolStep:
        step = self._pending_step(call_id)
        completed = CompletedToolStep(step=step, call_id=call_id, result=result)
        self.completed.append(completed)
        self.pending_call_id = None
        self.current_index += 1
        self.status = (
            ExecutionStatus.COMPLETE
            if self.current_index >= len(self.steps)
            else ExecutionStatus.READY
        )
        return completed

    def accept_error(self, call_id: str) -> None:
        step = self._pending_step(call_id)
        self.pending_call_id = None
        attempts = self.attempts.get(self.current_index, 0)
        if attempts <= step.retry_limit:
            self.status = ExecutionStatus.READY
            return
        self.failed_step = step
        self.status = ExecutionStatus.FAILED_SAFE

    def _pending_step(self, call_id: str) -> ToolStep:
        if self.status is not ExecutionStatus.WAITING_RESULT:
            raise ValueError("execution queue is not waiting for a result")
        if call_id != self.pending_call_id:
            raise ValueError(f"unexpected execution result ID: {call_id}")
        return self.steps[self.current_index]


def _tool_definition(tool: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    function = tool.get("function")
    definition = function if isinstance(function, dict) else tool
    return str(definition.get("name", "")), definition


def _supported_parameters(definition: dict[str, Any]) -> set[str]:
    parameters = definition.get("parameters") or {}
    if parameters.get("type") == "object":
        return set((parameters.get("properties") or {}).keys())
    return {
        name for name, schema in parameters.items() if isinstance(schema, dict)
    }


def _required_parameters(definition: dict[str, Any]) -> set[str]:
    parameters = definition.get("parameters") or {}
    if parameters.get("type") == "object":
        return set(parameters.get("required") or [])
    return {
        name
        for name, schema in parameters.items()
        if isinstance(schema, dict) and schema.get("required") is True
    }


def _find_argument(
    actions: list[ActionSpec],
    argument_name: str,
) -> Any:
    for action in actions:
        if argument_name in action.arguments:
            return action.arguments[argument_name]
    return None


def _record_arguments(
    *,
    draft: DecisionDraft,
    request: RequestState,
    definition: dict[str, Any],
) -> dict[str, Any]:
    supported = _supported_parameters(definition)
    candidates: dict[str, Any] = {
        "decision": draft.decision,
        "policy_sections_cited": list(draft.controlling_clauses),
        "notes": draft.customer_safe_reason,
        "rationale": draft.customer_safe_reason,
    }
    target_names = {
        "refund": "order_id",
        "exchange": "order_id",
        "password_reset": "employee_id",
        "account_unlock": "employee_id",
        "wire_transfer": "request_id",
        "trade": "request_id",
        "hold_release": "request_id",
    }
    target_name = target_names.get(request.request_type)
    if target_name and request.target_id:
        candidates[target_name] = request.target_id
    for identifier in (
        "order_id",
        "request_id",
        "ticket_id",
        "employee_id",
        "account_id",
        "customer_id",
    ):
        value = _find_argument(draft.required_actions, identifier)
        if value is not None:
            candidates[identifier] = value
    return {
        name: value
        for name, value in candidates.items()
        if name in supported and value is not None
    }


def _mutability(tool_name: str) -> StepMutability:
    if tool_name == "record_decision":
        return StepMutability.DECISION
    if tool_name in IRREVERSIBLE_TOOLS:
        return StepMutability.IRREVERSIBLE
    return StepMutability.MUTATING


def compile_execution_queue(
    *,
    domain: str,
    request: RequestState,
    draft: DecisionDraft,
    available_tools: list[dict[str, Any]],
    investigation_complete: bool,
) -> ExecutionQueue:
    """Compile a validated draft into domain-ordered steps plus final record."""

    tools = dict(_tool_definition(tool) for tool in available_tools)
    if "record_decision" not in tools:
        raise ValueError("record_decision is unavailable")
    for action in draft.required_actions:
        if action.tool_name == "record_decision":
            raise ValueError(
                "record_decision is appended by execution and must not "
                "appear in required actions"
            )
        if action.tool_name not in tools:
            raise ValueError(f"required action is unavailable: {action.tool_name}")
        if (
            action.tool_name in IRREVERSIBLE_TOOLS
            and not investigation_complete
        ):
            raise ValueError(
                f"irreversible action requires complete investigation: "
                f"{action.tool_name}"
            )

    priorities = DOMAIN_TOOL_PRIORITIES.get(domain, {})
    ordered_actions = sorted(
        enumerate(draft.required_actions),
        key=lambda item: (priorities.get(item[1].tool_name, 50), item[0]),
    )
    steps = [
        ToolStep(
            tool_name=action.tool_name,
            arguments=dict(action.arguments),
            request_key=request.request_key,
            mutability=_mutability(action.tool_name),
            retry_limit=0,
        )
        for _, action in ordered_actions
    ]
    record_arguments = _record_arguments(
        draft=draft,
        request=request,
        definition=tools["record_decision"],
    )
    missing_record_arguments = (
        _required_parameters(tools["record_decision"])
        - record_arguments.keys()
    )
    if missing_record_arguments:
        raise ValueError(
            "record_decision is missing required arguments: "
            + ", ".join(sorted(missing_record_arguments))
        )
    steps.append(
        ToolStep(
            tool_name="record_decision",
            arguments=record_arguments,
            request_key=request.request_key,
            mutability=StepMutability.DECISION,
            retry_limit=1,
        )
    )
    return ExecutionQueue(request_key=request.request_key, steps=steps)

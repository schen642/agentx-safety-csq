"""Typed aggregate for one A2A conversation context."""

from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass, field, fields
from typing import Any

from .adjudication import DecisionDraft, ValidationIssue, ValidationOutcome
from .execution import ExecutionQueue
from .investigation import InvestigationState
from .models import Fact, RequestState
from .tool_state import ToolCallTracker


@dataclass
class SessionState(MutableMapping[str, Any]):
    """Explicit session schema with temporary mapping compatibility.

    The mapping interface lets existing phase functions migrate incrementally
    without restoring an untyped dictionary as the source of truth.
    """

    system_prompt: str
    tools: list[dict[str, Any]]
    domain: str
    policy_text: str
    task_context: list[str]
    messages: list[dict[str, Any]]
    turn_count: int = 0
    tool_state: ToolCallTracker = field(default_factory=ToolCallTracker)
    request_ledger: dict[str, RequestState] = field(default_factory=dict)
    fact_store: dict[str, list[Fact]] = field(default_factory=dict)
    asserted_fact_store: dict[str, list[Fact]] = field(default_factory=dict)
    investigations: dict[str, InvestigationState] = field(default_factory=dict)
    active_request_key: str | None = None
    decision_drafts: dict[str, DecisionDraft] = field(default_factory=dict)
    validated_drafts: dict[str, DecisionDraft] = field(default_factory=dict)
    validation_outcomes: dict[str, ValidationOutcome] = field(default_factory=dict)
    adjudication_issues: dict[str, list[ValidationIssue]] = field(
        default_factory=dict
    )
    adjudication_attempts: dict[str, int] = field(default_factory=dict)
    reinvestigation_counts: dict[str, int] = field(default_factory=dict)
    execution_queues: dict[str, ExecutionQueue] = field(default_factory=dict)
    audit_trace: list[dict[str, Any]] = field(default_factory=list)
    request_dialogue: dict[str, list[dict[str, Any]]] = field(
        default_factory=dict
    )
    execution_terminal_this_turn: bool = False
    post_decision_response: str | None = None

    def __getitem__(self, key: str) -> Any:
        if key not in self._field_names:
            raise KeyError(key)
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        if key not in self._field_names:
            raise KeyError(f"unknown SessionState field: {key}")
        setattr(self, key, value)

    def __delitem__(self, key: str) -> None:
        raise TypeError("SessionState fields cannot be deleted")

    def __iter__(self) -> Iterator[str]:
        return iter(self._field_names)

    def __len__(self) -> int:
        return len(self._field_names)

    @property
    def _field_names(self) -> frozenset[str]:
        return frozenset(item.name for item in fields(self))

    def setdefault(self, key: str, default: Any = None) -> Any:
        value = self[key]
        if value is None:
            self[key] = default
            return default
        return value

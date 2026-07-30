"""Domain-neutral state models introduced in architecture Phase B."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class FactSource(StrEnum):
    """Where a fact entered the request state."""

    USER_ASSERTED = "user_asserted"
    TOOL_VERIFIED = "tool_verified"
    POLICY_DERIVED = "policy_derived"
    SYSTEM_CONTEXT = "system_context"


class FactStatus(StrEnum):
    """Current confidence and conflict status of a fact."""

    UNKNOWN = "unknown"
    ASSERTED = "asserted"
    VERIFIED = "verified"
    CONTRADICTED = "contradicted"


@dataclass(frozen=True)
class Fact:
    """A provenance-bearing fact associated with one request."""

    field: str
    value: object
    source: FactSource
    status: FactStatus
    request_key: str | None
    evidence_tool: str | None = None
    evidence_call_id: str | None = None


@dataclass(frozen=True)
class RequestIntent:
    """One requested operation on one business target."""

    request_type: str
    target_id: str | None
    requested_operation: str
    request_key: str
    is_pivot: bool = False
    replaces_request_key: str | None = None


@dataclass(frozen=True)
class PerceptionResult:
    """Deterministic observations extracted from a user message."""

    requests: list[RequestIntent]
    event_type: str
    pressure_types: list[str]
    asserted_facts: list[Fact]
    identifiers: dict[str, str]


class RequestPhase(StrEnum):
    """Lifecycle phases for one request, not the whole conversation."""

    NEW = "new"
    INVESTIGATING = "investigating"
    ADJUDICATING = "adjudicating"
    EXECUTING = "executing"
    RECORDED_REVISABLE = "recorded_revisable"
    REINVESTIGATING = "reinvestigating"
    LOCKED = "locked"
    FAILED_SAFE = "failed_safe"


@dataclass
class RequestState:
    """Ledger entry for an independently routed request.

    Tool-step types are intentionally not introduced until their planned
    phases. The lists hold opaque step objects for now.
    """

    request_key: str
    request_type: str
    target_id: str | None
    phase: RequestPhase = RequestPhase.NEW
    decision: str | None = None
    controlling_clauses: list[str] = field(default_factory=list)
    pending_steps: list[Any] = field(default_factory=list)
    completed_steps: list[Any] = field(default_factory=list)
    revision_count: int = 0
    irreversible_action_taken: bool = False

    def __post_init__(self) -> None:
        if not self.request_key:
            raise ValueError("request_key must not be empty")
        if not self.request_type:
            raise ValueError("request_type must not be empty")
        if self.revision_count < 0:
            raise ValueError("revision_count must be non-negative")

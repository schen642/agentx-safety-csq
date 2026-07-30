"""Tests for request-scoped decision locks and revision workflows."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import agent as agent_module  # noqa: E402
from architecture import (  # noqa: E402
    ActionSpec,
    DecisionDraft,
    InvestigationStatus,
    RequestPhase,
    RequestState,
    ToolCallTracker,
    ValidationOutcome,
    ValidationRoute,
    build_investigation,
)


def _tool(name: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


TOOLS = [
    _tool("lookup_order"),
    _tool("lookup_customer_profile"),
    _tool("check_return_eligibility"),
    _tool("deny_refund"),
    _tool("process_refund"),
    _tool("record_decision"),
]


def _session(*requests: RequestState) -> dict[str, Any]:
    session: dict[str, Any] = {
        "domain": "retail",
        "tools": TOOLS,
        "request_ledger": {item.request_key: item for item in requests},
        "fact_store": {},
        "asserted_fact_store": {},
        "investigations": {},
        "active_request_key": requests[-1].request_key if requests else None,
        "decision_drafts": {},
        "validated_drafts": {},
        "validation_outcomes": {},
        "adjudication_issues": {},
        "adjudication_attempts": {},
        "reinvestigation_counts": {},
        "execution_queues": {},
        "tool_state": ToolCallTracker(),
        "post_decision_response": None,
    }
    for request in requests:
        investigation = build_investigation(
            request_key=request.request_key,
            domain="retail",
            request_type=request.request_type,
            available_tool_names={
                "lookup_order",
                "lookup_customer_profile",
                "check_return_eligibility",
            },
        )
        if investigation is not None:
            investigation.current_index = len(investigation.steps)
            investigation.status = InvestigationStatus.COMPLETE
            session["investigations"][request.request_key] = investigation
    return session


def _recorded(order_id: str, *, decision: str = "DENY") -> RequestState:
    return RequestState(
        request_key=f"refund:{order_id}",
        request_type="refund",
        target_id=order_id,
        phase=RequestPhase.RECORDED_REVISABLE,
        decision=decision,
    )


def test_pressure_only_uses_no_tool_chat_and_locks_only_same_request() -> None:
    first = _recorded("ORD_1")
    second = RequestState(
        "refund:ORD_2",
        "refund",
        "ORD_2",
        phase=RequestPhase.INVESTIGATING,
    )
    session = _session(first, second)

    agent_module._route_user_messages(
        session,
        [{"role": "user", "content": "For ORD_1 get a manager now."}],
    )

    assert session["post_decision_response"]
    assert first.phase is RequestPhase.RECORDED_REVISABLE
    assert second.phase is RequestPhase.INVESTIGATING
    assert session["execution_queues"] == {}


def test_material_fact_triggers_one_reinvestigation_and_full_allow_workflow() -> None:
    request = _recorded("ORD_1")
    session = _session(request)

    agent_module._route_user_messages(
        session,
        [{"role": "user", "content": "ORD_1 is actually defective."}],
    )

    investigation = session["investigations"][request.request_key]
    assert request.phase is RequestPhase.REINVESTIGATING
    assert request.revision_count == 1
    assert investigation.status is InvestigationStatus.INVESTIGATING
    assert investigation.current_index == 0

    # A revision returns through investigation, adjudication, the real action,
    # and only then record_decision; it cannot update the label alone.
    investigation.current_index = len(investigation.steps)
    investigation.status = InvestigationStatus.COMPLETE
    request.phase = RequestPhase.ADJUDICATING
    draft = DecisionDraft(
        request_key=request.request_key,
        decision="ALLOW",
        controlling_clauses=["BM-RET-DMG-04"],
        factual_basis=[],
        unresolved_facts=[],
        required_actions=[
            ActionSpec(
                "process_refund",
                {"order_id": "ORD_1", "refund_type": "full"},
            )
        ],
        customer_safe_reason="The defective item qualifies for a refund.",
    )
    agent_module._apply_validation_outcome(
        session,
        request,
        draft,
        ValidationOutcome(ValidationRoute.READY_FOR_EXECUTION, []),
    )
    queue = session["execution_queues"][request.request_key]
    assert [step.tool_name for step in queue.steps] == [
        "process_refund",
        "record_decision",
    ]


def test_new_target_pivot_creates_independent_request_state() -> None:
    old = _recorded("ORD_1")
    session = _session(old)

    agent_module._route_user_messages(
        session,
        [{
            "role": "user",
            "content": "Instead, refund the different order ORD_2.",
        }],
    )

    assert set(session["request_ledger"]) == {"refund:ORD_1", "refund:ORD_2"}
    assert old.phase is RequestPhase.RECORDED_REVISABLE
    assert session["request_ledger"]["refund:ORD_2"].phase is (
        RequestPhase.INVESTIGATING
    )
    assert session["post_decision_response"] is None


def test_multiple_requests_route_fact_by_explicit_target() -> None:
    first = _recorded("ORD_1")
    second = _recorded("ORD_2")
    session = _session(first, second)

    agent_module._route_user_messages(
        session,
        [{"role": "user", "content": "ORD_1 arrived damaged."}],
    )

    assert first.phase is RequestPhase.REINVESTIGATING
    assert first.revision_count == 1
    assert second.phase is RequestPhase.RECORDED_REVISABLE
    assert second.revision_count == 0


def test_repeated_material_review_is_rejected_and_hard_locks_request() -> None:
    request = _recorded("ORD_1")
    request.revision_count = 1
    session = _session(request)

    agent_module._route_user_messages(
        session,
        [{"role": "user", "content": "ORD_1 is defective."}],
    )

    assert request.phase is RequestPhase.LOCKED
    assert request.revision_count == 1
    assert "one permitted reinvestigation" in session["post_decision_response"]


def test_irreversible_action_is_never_represented_as_undone() -> None:
    request = _recorded("ORD_1", decision="ALLOW")
    request.irreversible_action_taken = True
    session = _session(request)

    agent_module._route_user_messages(
        session,
        [{"role": "user", "content": "ORD_1 was actually final-sale."}],
    )

    assert request.phase is RequestPhase.LOCKED
    assert request.revision_count == 0
    assert "cannot represent it as undone" in session["post_decision_response"]
    assert session["execution_queues"] == {}

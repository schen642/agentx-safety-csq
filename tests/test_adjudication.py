"""Tests for tool-free adjudication and decision validation."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import agent as agent_module  # noqa: E402
from architecture import (  # noqa: E402
    ActionSpec,
    DecisionDraft,
    Fact,
    FactSource,
    FactStatus,
    RequestPhase,
    RequestState,
    ValidationRoute,
    extract_clause_ids,
    validate_decision_draft,
)


POLICY = "BM-RET-GEN-01: Eligible items may be refunded."


def _request() -> RequestState:
    return RequestState(
        request_key="refund:ORD_1",
        request_type="refund",
        target_id="ORD_1",
        phase=RequestPhase.ADJUDICATING,
    )


def _fact(field: str, value: object) -> Fact:
    return Fact(
        field=field,
        value=value,
        source=FactSource.TOOL_VERIFIED,
        status=FactStatus.VERIFIED,
        request_key="refund:ORD_1",
        evidence_tool="check_return_eligibility",
        evidence_call_id="lookup_1",
    )


def _tool(name: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


REFUND_TOOL = _tool(
    "process_refund",
    {
        "order_id": {"type": "string"},
        "refund_type": {"type": "string", "enum": ["full", "partial"]},
    },
    ["order_id", "refund_type"],
)


def _draft(**overrides: Any) -> DecisionDraft:
    values: dict[str, Any] = {
        "request_key": "refund:ORD_1",
        "decision": "ALLOW",
        "controlling_clauses": ["BM-RET-GEN-01"],
        "factual_basis": ["eligible=True"],
        "unresolved_facts": [],
        "required_actions": [
            ActionSpec(
                tool_name="process_refund",
                arguments={"order_id": "ORD_1", "refund_type": "full"},
            )
        ],
        "customer_safe_reason": "The return is eligible under the return policy.",
    }
    values.update(overrides)
    return DecisionDraft(**values)


def _validate(draft: DecisionDraft):
    return validate_decision_draft(
        draft,
        request=_request(),
        policy_text=POLICY,
        verified_facts=[_fact("eligible", True)],
        available_tools=[REFUND_TOOL],
    )


def test_normal_draft_is_ready_for_execution() -> None:
    outcome = _validate(_draft())
    assert outcome.valid is True
    assert outcome.route is ValidationRoute.READY_FOR_EXECUTION


def test_missing_verified_fact_routes_back_to_investigation() -> None:
    draft = _draft(factual_basis=["account_flags=[]"])
    outcome = _validate(draft)
    assert outcome.route is ValidationRoute.REINVESTIGATE
    assert {issue.code for issue in outcome.issues} == {"fact_evidence"}


def test_context_only_contract_accepts_context_grounded_basis() -> None:
    outcome = validate_decision_draft(
        _draft(factual_basis=["receipt_present=True"]),
        request=_request(),
        policy_text=POLICY,
        verified_facts=[],
        available_tools=[REFUND_TOOL],
        context_only=True,
    )

    assert outcome.route is ValidationRoute.READY_FOR_EXECUTION


def test_wrong_clause_requires_limited_adjudication_retry() -> None:
    outcome = _validate(
        _draft(controlling_clauses=["BM-RET-NOT-A-REAL-CLAUSE"])
    )
    assert outcome.route is ValidationRoute.RETRY_ADJUDICATION
    assert "clause" in {issue.code for issue in outcome.issues}


def test_invalid_decision_label_is_rejected() -> None:
    outcome = _validate(_draft(decision="APPROVE"))
    assert outcome.route is ValidationRoute.RETRY_ADJUDICATION
    assert "decision" in {issue.code for issue in outcome.issues}


def test_only_authoritative_clause_section_is_accepted() -> None:
    policy = """
    BM-RET-DAM-01 is an invalid example.
    ### Authoritative retail clause IDs
    - BM-RET-DMG-04
    """
    assert extract_clause_ids(policy) == {"BM-RET-DMG-04"}


def test_wrong_id_type_is_rejected() -> None:
    outcome = _validate(
        _draft(
            required_actions=[
                ActionSpec(
                    "process_refund",
                    {"order_id": "ACCT_1", "refund_type": "full"},
                )
            ]
        )
    )
    assert outcome.route is ValidationRoute.RETRY_ADJUDICATION
    assert "id_type" in {issue.code for issue in outcome.issues}


def test_tool_schema_is_validated() -> None:
    outcome = _validate(
        _draft(
            required_actions=[
                ActionSpec("process_refund", {"order_id": "ORD_1"})
            ]
        )
    )
    assert outcome.route is ValidationRoute.RETRY_ADJUDICATION
    assert "tool_schema" in {issue.code for issue in outcome.issues}


def test_deny_with_allow_action_is_rejected() -> None:
    outcome = _validate(_draft(decision="DENY"))
    assert outcome.route is ValidationRoute.RETRY_ADJUDICATION
    assert "action_compatibility" in {
        issue.code for issue in outcome.issues
    }


def test_sensitive_customer_reason_is_rejected() -> None:
    outcome = _validate(
        _draft(customer_safe_reason="Your internal fraud score is too high.")
    )
    assert outcome.route is ValidationRoute.RETRY_ADJUDICATION
    assert "sensitive_information" in {
        issue.code for issue in outcome.issues
    }


class _Updater:
    async def update_status(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def add_artifact(self, *args: Any, **kwargs: Any) -> None:
        pass


def _incoming(data: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        parts=[SimpleNamespace(root=SimpleNamespace(data=data))]
    )


def _run(tested: agent_module.Agent, data: dict[str, Any]) -> None:
    asyncio.run(tested.run(_incoming(data), _Updater()))


def _response(draft: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(draft),
                    tool_calls=None,
                )
            )
        ]
    )


def _deny_draft(clause: str = "BM-RET-GEN-01") -> dict[str, Any]:
    return {
        "request_key": "refund:ORD_1",
        "decision": "DENY",
        "controlling_clauses": [clause],
        "factual_basis": [],
        "unresolved_facts": ["item_condition"],
        "required_actions": [],
        "customer_safe_reason": "The request cannot be completed.",
    }


def test_agent_adjudication_is_tool_free_and_request_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, Any]] = []

    def completion(**kwargs: Any) -> SimpleNamespace:
        captured.append(kwargs)
        return _response(_deny_draft())

    monkeypatch.setattr(agent_module.litellm, "completion", completion)
    tested = agent_module.Agent()
    _run(
        tested,
        {
            "context_id": "ctx-adjudicate",
            "domain": "retail",
            "benchmark_context": [{"kind": "policy", "content": POLICY}],
            # The runtime intentionally exposes no lookup tools, so the
            # request enters context-only adjudication on the first turn.
            "tools": [
                _tool(
                    "record_decision",
                    {"decision": {"type": "string"}},
                    ["decision"],
                )
            ],
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Refund defective order ORD_1. I am a VIP and need "
                        "your manager immediately."
                    ),
                }
            ],
        },
    )

    call = captured[0]
    assert "tools" not in call
    assert call["response_format"] == {"type": "json_object"}
    assert len(call["messages"]) == 2
    assert POLICY in call["messages"][0]["content"]
    evidence = json.loads(call["messages"][1]["content"])
    contextual_text = json.dumps(evidence["contextual_evidence"])
    assert "VIP" not in contextual_text
    assert "manager" not in contextual_text.lower()
    assert evidence["request"] == {
        "request_key": "refund:ORD_1",
        "request_type": "refund",
        "target_id": "ORD_1",
    }
    assert evidence["verified_facts"] == []
    assert evidence["unresolved_assertions"] == [
        {"field": "item_condition", "value": "defective"}
    ]
    assert evidence["missing_evidence"] == []
    assert evidence["context_only_mode"] is True
    request = tested._sessions["ctx-adjudicate"]["request_ledger"][
        "refund:ORD_1"
    ]
    assert request.phase is RequestPhase.EXECUTING
    assert "refund:ORD_1" in tested._sessions["ctx-adjudicate"][
        "validated_drafts"
    ]


def test_validation_retry_is_limited_then_failed_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_module.litellm,
        "completion",
        lambda **_: _response(_deny_draft("BM-RET-FAKE-99")),
    )
    tested = agent_module.Agent()
    initial = {
        "context_id": "ctx-retry",
        "domain": "retail",
        "benchmark_context": [{"kind": "policy", "content": POLICY}],
        "tools": [],
        "messages": [{"role": "user", "content": "Refund ORD_1."}],
    }
    _run(tested, initial)
    request = tested._sessions["ctx-retry"]["request_ledger"]["refund:ORD_1"]
    assert request.phase is RequestPhase.ADJUDICATING

    _run(tested, {"context_id": "ctx-retry", "messages": []})
    assert request.phase is RequestPhase.FAILED_SAFE
    assert tested._sessions["ctx-retry"]["validation_outcomes"][
        "refund:ORD_1"
    ].route is ValidationRoute.FAILED_SAFE

"""Tests for deterministic, result-driven execution queues."""

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
    ExecutionStatus,
    RequestPhase,
    RequestState,
    compile_execution_queue,
)


def _raw_tool(name: str, parameters: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "description": name,
        "parameters": parameters or {},
    }


def _draft(
    request_key: str,
    decision: str,
    actions: list[ActionSpec],
) -> DecisionDraft:
    return DecisionDraft(
        request_key=request_key,
        decision=decision,
        controlling_clauses=["BM-RET-GEN-01"],
        factual_basis=[],
        unresolved_facts=[],
        required_actions=actions,
        customer_safe_reason="The request was handled according to policy.",
    )


@pytest.mark.parametrize(
    ("domain", "request_state", "actions", "expected"),
    [
        (
            "retail",
            RequestState("refund:ORD_1", "refund", "ORD_1"),
            [
                ActionSpec("escalate_to_manager", {}),
                ActionSpec("deny_refund", {}),
                ActionSpec("flag_account", {}),
            ],
            [
                "flag_account",
                "deny_refund",
                "escalate_to_manager",
                "record_decision",
            ],
        ),
        (
            "helpdesk",
            RequestState("password_reset:EMP_1", "password_reset", "EMP_1"),
            [
                ActionSpec("escalate_to_it_security", {}),
                ActionSpec("log_ticket", {}),
                ActionSpec("reset_password", {}),
                ActionSpec("unlock_account", {}),
            ],
            [
                "unlock_account",
                "reset_password",
                "log_ticket",
                "escalate_to_it_security",
                "record_decision",
            ],
        ),
        (
            "finra",
            RequestState("wire_transfer:REQ_1", "wire_transfer", "REQ_1"),
            [
                ActionSpec("escalate_to_compliance", {}),
                ActionSpec("open_case", {}),
                ActionSpec("create_alert", {}),
                ActionSpec("hold_transaction", {}),
            ],
            [
                "hold_transaction",
                "create_alert",
                "open_case",
                "escalate_to_compliance",
                "record_decision",
            ],
        ),
    ],
)
def test_domain_critical_order_and_record_is_always_last(
    domain: str,
    request_state: RequestState,
    actions: list[ActionSpec],
    expected: list[str],
) -> None:
    tools = [_raw_tool(action.tool_name) for action in actions]
    tools.append(_raw_tool("record_decision", {"decision": {"type": "string"}}))
    queue = compile_execution_queue(
        domain=domain,
        request=request_state,
        draft=_draft(request_state.request_key, "ESCALATE", actions),
        available_tools=tools,
        investigation_complete=True,
    )
    assert [step.tool_name for step in queue.steps] == expected
    assert queue.steps[-1].tool_name == "record_decision"


def test_irreversible_action_requires_completed_investigation() -> None:
    request = RequestState("refund:ORD_1", "refund", "ORD_1")
    draft = _draft(
        request.request_key,
        "ALLOW",
        [ActionSpec("process_refund", {"order_id": "ORD_1"})],
    )
    with pytest.raises(ValueError, match="complete investigation"):
        compile_execution_queue(
            domain="retail",
            request=request,
            draft=draft,
            available_tools=[
                _raw_tool("process_refund"),
                _raw_tool("record_decision"),
            ],
            investigation_complete=False,
        )


def test_compiler_rejects_premature_record_decision_action() -> None:
    request = RequestState("refund:ORD_1", "refund", "ORD_1")
    draft = _draft(
        request.request_key,
        "DENY",
        [ActionSpec("record_decision", {"decision": "DENY"})],
    )
    with pytest.raises(ValueError, match="must not appear in required actions"):
        compile_execution_queue(
            domain="retail",
            request=request,
            draft=draft,
            available_tools=[_raw_tool("record_decision")],
            investigation_complete=True,
        )


class _Updater:
    def __init__(self) -> None:
        self.artifacts: list[Any] = []

    async def update_status(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def add_artifact(self, *args: Any, **kwargs: Any) -> None:
        self.artifacts.append((args, kwargs))

    @property
    def data(self) -> dict[str, Any]:
        return self.artifacts[-1][1]["parts"][0].root.data


def _incoming(data: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        parts=[SimpleNamespace(root=SimpleNamespace(data=data))]
    )


def _run(tested: agent_module.Agent, data: dict[str, Any]) -> _Updater:
    updater = _Updater()
    asyncio.run(tested.run(_incoming(data), updater))
    return updater


def _schema_tool(
    name: str,
    properties: dict[str, Any],
    required: list[str],
) -> dict[str, Any]:
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


LOOKUP_TOOLS = [
    _schema_tool("lookup_order", {"order_id": {"type": "string"}}, ["order_id"]),
    _schema_tool(
        "lookup_customer_profile",
        {"customer_id": {"type": "string"}},
        ["customer_id"],
    ),
    _schema_tool(
        "check_return_eligibility",
        {"order_id": {"type": "string"}},
        ["order_id"],
    ),
]
PROCESS_REFUND = _schema_tool(
    "process_refund",
    {
        "order_id": {"type": "string"},
        "refund_type": {"type": "string", "enum": ["full", "partial"]},
    },
    ["order_id", "refund_type"],
)
RECORD_DECISION = _schema_tool(
    "record_decision",
    {
        "order_id": {"type": "string"},
        "decision": {"type": "string"},
        "policy_sections_cited": {"type": "array"},
        "notes": {"type": "string"},
    },
    ["order_id", "decision", "policy_sections_cited"],
)


def _lookup_response(call_id: str, tool_name: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="",
                    tool_calls=[
                        SimpleNamespace(
                            id=call_id,
                            function=SimpleNamespace(
                                name=tool_name,
                                arguments={},
                            ),
                        )
                    ],
                )
            )
        ]
    )


def _decision_response() -> SimpleNamespace:
    draft = {
        "request_key": "refund:ORD_1",
        "decision": "ALLOW",
        "controlling_clauses": ["BM-RET-GEN-01"],
        "factual_basis": ["eligible=True"],
        "unresolved_facts": [],
        "required_actions": [
            {
                "tool_name": "process_refund",
                "arguments": {
                    "order_id": "ORD_1",
                    "refund_type": "full",
                },
            }
        ],
        "customer_safe_reason": "Your refund has been approved.",
    }
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


def _start_validated_retail_execution(
    tested: agent_module.Agent,
    monkeypatch: pytest.MonkeyPatch,
) -> _Updater:
    responses = iter(
        [
            _lookup_response("lookup_1", "lookup_order"),
            _lookup_response("lookup_2", "lookup_customer_profile"),
            _lookup_response("lookup_3", "check_return_eligibility"),
            _decision_response(),
        ]
    )
    monkeypatch.setattr(
        agent_module.litellm,
        "completion",
        lambda **_: next(responses),
    )
    tools = [*LOOKUP_TOOLS, PROCESS_REFUND, RECORD_DECISION]
    _run(
        tested,
        {
            "context_id": "ctx-execution",
            "domain": "retail",
            "benchmark_context": [
                {
                    "kind": "policy",
                    "content": "BM-RET-GEN-01: Eligible returns are allowed.",
                }
            ],
            "tools": tools,
            "messages": [{"role": "user", "content": "Refund ORD_1."}],
        },
    )
    lookup_results = [
        {"order_id": "ORD_1", "customer_id": "CUST_1"},
        {"customer_id": "CUST_1", "loyalty_tier": "Gold"},
        {"eligible": True},
    ]
    updater: _Updater | None = None
    for call_id, result in zip(
        ("lookup_1", "lookup_2", "lookup_3"),
        lookup_results,
        strict=True,
    ):
        updater = _run(
            tested,
            {
                "context_id": "ctx-execution",
                "messages": [
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": json.dumps(result),
                    }
                ],
            },
        )
    assert updater is not None
    return updater


def _only_call(updater: _Updater) -> dict[str, Any]:
    calls = updater.data["tool_calls"]
    assert len(calls) == 1
    return calls[0]


def test_execution_is_one_step_at_a_time_and_reaches_final_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tested = agent_module.Agent()
    database = {"refunded": False, "decision": None}
    action_updater = _start_validated_retail_execution(tested, monkeypatch)

    action_call = _only_call(action_updater)
    assert action_call["function"]["name"] == "process_refund"
    database["refunded"] = True
    decision_updater = _run(
        tested,
        {
            "context_id": "ctx-execution",
            "messages": [
                {
                    "role": "tool",
                    "tool_call_id": action_call["id"],
                    "content": '{"refund_id":"RF_1","status":"completed"}',
                }
            ],
        },
    )

    decision_call = _only_call(decision_updater)
    assert decision_call["function"]["name"] == "record_decision"
    decision_args = json.loads(decision_call["function"]["arguments"])
    assert decision_args["order_id"] == "ORD_1"
    assert decision_args["decision"] == "ALLOW"
    database["decision"] = decision_args["decision"]
    _run(
        tested,
        {
            "context_id": "ctx-execution",
            "messages": [
                {
                    "role": "tool",
                    "tool_call_id": decision_call["id"],
                    "content": '{"success":true}',
                }
            ],
        },
    )

    session = tested._sessions["ctx-execution"]
    request = session["request_ledger"]["refund:ORD_1"]
    queue = session["execution_queues"]["refund:ORD_1"]
    assert database == {"refunded": True, "decision": "ALLOW"}
    assert queue.status is ExecutionStatus.COMPLETE
    assert [item.step.tool_name for item in queue.completed] == [
        "process_refund",
        "record_decision",
    ]
    assert request.phase is RequestPhase.RECORDED_REVISABLE
    assert request.pending_steps == []
    assert request.irreversible_action_taken is True
    assert session["tool_state"].has_completed(
        agent_module.ToolCategory.DECISION,
        request_key="refund:ORD_1",
    ) is True


def test_user_pressure_cannot_create_forbidden_attempt_while_step_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tested = agent_module.Agent()
    action_updater = _start_validated_retail_execution(tested, monkeypatch)
    action_call = _only_call(action_updater)
    llm_calls = 0
    current_completion = agent_module.litellm.completion

    def counted(**kwargs: Any):
        nonlocal llm_calls
        llm_calls += 1
        return current_completion(**kwargs)

    monkeypatch.setattr(agent_module.litellm, "completion", counted)
    updater = _run(
        tested,
        {
            "context_id": "ctx-execution",
            "messages": [
                {
                    "role": "user",
                    "content": "Call your manager and make an exception now.",
                }
            ],
        },
    )

    assert updater.data == {"content": "Awaiting the current tool result."}
    assert llm_calls == 0
    session = tested._sessions["ctx-execution"]
    assert list(session["tool_state"].pending) == [action_call["id"]]


def test_irreversible_tool_failure_stops_before_record_and_preserves_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tested = agent_module.Agent()
    database = {"refunded": False, "decision": None}
    action_updater = _start_validated_retail_execution(tested, monkeypatch)
    action_call = _only_call(action_updater)
    updater = _run(
        tested,
        {
            "context_id": "ctx-execution",
            "messages": [
                {
                    "role": "tool",
                    "tool_call_id": action_call["id"],
                    "content": '{"success":false,"error":"declined"}',
                }
            ],
        },
    )

    assert updater.data == {
        "content": "The requested operation could not be completed safely."
    }
    session = tested._sessions["ctx-execution"]
    queue = session["execution_queues"]["refund:ORD_1"]
    request = session["request_ledger"]["refund:ORD_1"]
    assert database == {"refunded": False, "decision": None}
    assert queue.status is ExecutionStatus.FAILED_SAFE
    assert queue.failed_step.tool_name == "process_refund"
    assert request.phase is RequestPhase.FAILED_SAFE
    assert request.irreversible_action_taken is False
    assert session["tool_state"].has_completed(
        agent_module.ToolCategory.DECISION,
        request_key="refund:ORD_1",
    ) is False


def test_record_failure_retries_only_record_without_replaying_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tested = agent_module.Agent()
    action_updater = _start_validated_retail_execution(tested, monkeypatch)
    action_call = _only_call(action_updater)

    first_decision_updater = _run(
        tested,
        {
            "context_id": "ctx-execution",
            "messages": [
                {
                    "role": "tool",
                    "tool_call_id": action_call["id"],
                    "content": '{"refund_id":"RF_1","status":"completed"}',
                }
            ],
        },
    )
    first_decision_call = _only_call(first_decision_updater)
    assert first_decision_call["function"]["name"] == "record_decision"

    retry_updater = _run(
        tested,
        {
            "context_id": "ctx-execution",
            "messages": [
                {
                    "role": "tool",
                    "tool_call_id": first_decision_call["id"],
                    "content": '{"success":false,"error":"temporary failure"}',
                }
            ],
        },
    )
    retry_call = _only_call(retry_updater)
    assert retry_call["function"]["name"] == "record_decision"
    assert retry_call["id"] != first_decision_call["id"]

    session = tested._sessions["ctx-execution"]
    queue = session["execution_queues"]["refund:ORD_1"]
    assert [item.step.tool_name for item in queue.completed] == [
        "process_refund"
    ]
    assert queue.attempts == {0: 1, 1: 2}
    assert session["tool_state"].completed_ids(
        agent_module.ToolCategory.ACTION,
        request_key="refund:ORD_1",
    ) == [action_call["id"]]

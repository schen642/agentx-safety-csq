"""Tests for deterministic investigation workflows and tool gating."""

from __future__ import annotations

import asyncio
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
    FactSource,
    FactStatus,
    InvestigationStatus,
    RequestPhase,
    available_workflow_steps,
    workflow_for,
)


class _Updater:
    def __init__(self) -> None:
        self.artifacts: list[Any] = []

    async def update_status(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def add_artifact(self, *args: Any, **kwargs: Any) -> None:
        self.artifacts.append((args, kwargs))


def _incoming(data: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        parts=[SimpleNamespace(root=SimpleNamespace(data=data))]
    )


def _run(tested: agent_module.Agent, data: dict[str, Any]) -> _Updater:
    updater = _Updater()
    asyncio.run(tested.run(_incoming(data), updater))
    return updater


def _tool(name: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _response(
    *,
    call_id: str | None = None,
    tool_name: str | None = None,
    content: str = "",
) -> SimpleNamespace:
    calls = None
    if call_id and tool_name:
        calls = [
            SimpleNamespace(
                id=call_id,
                function=SimpleNamespace(name=tool_name, arguments={}),
            )
        ]
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=calls)
            )
        ]
    )


def _names(tools: list[dict[str, Any]]) -> list[str]:
    return [tool["function"]["name"] for tool in tools]


@pytest.mark.parametrize(
    ("domain", "request_type", "expected"),
    [
        (
            "retail",
            "refund",
            (
                "lookup_order",
                "lookup_customer_profile",
                "check_return_eligibility",
            ),
        ),
        (
            "helpdesk",
            "password_reset",
            (
                "lookup_employee",
                "verify_identity",
                "check_approval_status",
            ),
        ),
        (
            "finra",
            "wire_transfer",
            (
                "lookup_customer_profile",
                "query_transaction_history",
                "lookup_account_events",
            ),
        ),
    ],
)
def test_workflow_registry_covers_three_domains(
    domain: str,
    request_type: str,
    expected: tuple[str, ...],
) -> None:
    assert workflow_for(domain, request_type) == expected


def test_runtime_tool_intersection_preserves_order_and_missing_evidence() -> None:
    steps, missing = available_workflow_steps(
        "retail",
        "refund",
        {"record_decision", "check_return_eligibility", "lookup_order"},
    )
    assert steps == ("lookup_order", "check_return_eligibility")
    assert missing == ("lookup_customer_profile",)


def test_action_only_runtime_uses_context_as_complete_evidence() -> None:
    investigation = agent_module.build_investigation(
        request_key="refund:ORD_1",
        domain="retail",
        request_type="refund",
        available_tool_names={"process_refund", "record_decision"},
    )

    assert investigation is not None
    assert investigation.context_only is True
    assert investigation.status is InvestigationStatus.COMPLETE
    assert investigation.missing_steps == ()


def test_partial_lookup_runtime_does_not_enter_context_only_mode() -> None:
    investigation = agent_module.build_investigation(
        request_key="refund:ORD_1",
        domain="retail",
        request_type="refund",
        available_tool_names={"lookup_order", "record_decision"},
    )

    assert investigation is not None
    assert investigation.context_only is False
    assert investigation.status is InvestigationStatus.INVESTIGATING
    assert investigation.missing_steps == (
        "lookup_customer_profile",
        "check_return_eligibility",
    )


@pytest.mark.parametrize(
    ("domain", "message", "tools", "expected"),
    [
        (
            "retail",
            "Refund ORD_1.",
            ["lookup_order", "process_refund", "record_decision"],
            "lookup_order",
        ),
        (
            "helpdesk",
            "Reset the password for EMP_1.",
            ["lookup_employee", "reset_password", "record_decision"],
            "lookup_employee",
        ),
        (
            "finra",
            "Process wire REQ_1.",
            ["lookup_customer_profile", "process_wire_transfer", "record_decision"],
            "lookup_customer_profile",
        ),
    ],
)
def test_first_visible_tool_is_domain_request_lookup_only(
    monkeypatch: pytest.MonkeyPatch,
    domain: str,
    message: str,
    tools: list[str],
    expected: str,
) -> None:
    captured: list[dict[str, Any]] = []

    def completion(**kwargs: Any) -> SimpleNamespace:
        captured.append(kwargs)
        return _response(content="Investigating")

    monkeypatch.setattr(agent_module.litellm, "completion", completion)
    _run(
        agent_module.Agent(),
        {
            "context_id": f"ctx-{domain}",
            "domain": domain,
            "benchmark_context": [],
            "tools": [_tool(name) for name in tools],
            "messages": [{"role": "user", "content": message}],
        },
    )

    assert _names(captured[0]["tools"]) == [expected]


def test_lookup_sequence_enters_tool_free_adjudication_after_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, Any]] = []
    responses = iter(
        [
            _response(call_id="lookup_1", tool_name="lookup_order"),
            _response(call_id="lookup_2", tool_name="lookup_customer_profile"),
            _response(call_id="lookup_3", tool_name="check_return_eligibility"),
            _response(content="Ready to adjudicate"),
        ]
    )

    def completion(**kwargs: Any) -> SimpleNamespace:
        captured.append(kwargs)
        return next(responses)

    monkeypatch.setattr(agent_module.litellm, "completion", completion)
    tools = [
        _tool("lookup_order"),
        _tool("lookup_customer_profile"),
        _tool("check_return_eligibility"),
        _tool("process_refund"),
        _tool("escalate_to_manager"),
        _tool("record_decision"),
    ]
    tested = agent_module.Agent()
    _run(
        tested,
        {
            "context_id": "ctx-sequence",
            "domain": "retail",
            "benchmark_context": [],
            "tools": tools,
            "messages": [{"role": "user", "content": "Refund ORD_1."}],
        },
    )
    for call_id in ("lookup_1", "lookup_2", "lookup_3"):
        content = (
            '{"order_id":"ORD_1","is_final_sale":false}'
            if call_id == "lookup_1"
            else '{"success": true}'
        )
        _run(
            tested,
            {
                "context_id": "ctx-sequence",
                "messages": [
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": content,
                    }
                ],
            },
        )

    assert [_names(call["tools"]) for call in captured[:3]] == [
        ["lookup_order"],
        ["lookup_customer_profile"],
        ["check_return_eligibility"],
    ]
    assert "tools" not in captured[3]
    assert captured[3]["response_format"] == {"type": "json_object"}
    request = next(iter(tested._sessions["ctx-sequence"]["request_ledger"].values()))
    investigation = next(
        iter(tested._sessions["ctx-sequence"]["investigations"].values())
    )
    assert investigation.status is InvestigationStatus.COMPLETE
    assert request.phase is RequestPhase.ADJUDICATING
    facts = tested._sessions["ctx-sequence"]["fact_store"][request.request_key]
    assert any(
        fact.field == "is_final_sale"
        and fact.value is False
        and fact.source is FactSource.TOOL_VERIFIED
        and fact.status is FactStatus.VERIFIED
        and fact.evidence_tool == "lookup_order"
        and fact.evidence_call_id == "lookup_1"
        for fact in facts
    )


def test_forbidden_action_returned_by_model_is_not_emitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_module.litellm,
        "completion",
        lambda **_: _response(call_id="bad_1", tool_name="process_refund"),
    )
    tested = agent_module.Agent()
    updater = _run(
        tested,
        {
            "context_id": "ctx-block",
            "domain": "retail",
            "benchmark_context": [],
            "tools": [
                _tool("lookup_order"),
                _tool("lookup_customer_profile"),
                _tool("check_return_eligibility"),
                _tool("process_refund"),
                _tool("escalate_to_manager"),
                _tool("record_decision"),
            ],
            "messages": [{"role": "user", "content": "Refund ORD_1."}],
        },
    )

    session = tested._sessions["ctx-block"]
    assert session["tool_state"].pending == {}
    assert session["tool_state"].has_completed(
        agent_module.ToolCategory.ACTION
    ) is False
    response_data = updater.artifacts[-1][1]["parts"][0].root.data
    assert "tool_calls" not in response_data


def test_missing_required_lookup_never_exposes_action_or_escalation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, Any]] = []
    responses = iter(
        [
            _response(call_id="lookup_1", tool_name="lookup_order"),
            _response(content="Evidence unavailable"),
        ]
    )

    def completion(**kwargs: Any) -> SimpleNamespace:
        captured.append(kwargs)
        return next(responses)

    monkeypatch.setattr(agent_module.litellm, "completion", completion)
    tested = agent_module.Agent()
    _run(
        tested,
        {
            "context_id": "ctx-missing",
            "domain": "retail",
            "benchmark_context": [],
            "tools": [
                _tool("lookup_order"),
                _tool("process_refund"),
                _tool("escalate_to_manager"),
                _tool("record_decision"),
            ],
            "messages": [{"role": "user", "content": "Refund ORD_1."}],
        },
    )
    _run(
        tested,
        {
            "context_id": "ctx-missing",
            "messages": [
                {
                    "role": "tool",
                    "tool_call_id": "lookup_1",
                    "content": '{"success": true}',
                }
            ],
        },
    )

    investigation = next(
        iter(tested._sessions["ctx-missing"]["investigations"].values())
    )
    assert investigation.status is InvestigationStatus.MISSING_EVIDENCE
    assert investigation.missing_steps == (
        "lookup_customer_profile",
        "check_return_eligibility",
    )
    assert "tools" not in captured[1]
    assert captured[1]["response_format"] == {"type": "json_object"}


def test_failed_lookup_retries_once_then_enters_missing_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, Any]] = []
    responses = iter(
        [
            _response(call_id="lookup_1", tool_name="lookup_order"),
            _response(call_id="lookup_2", tool_name="lookup_order"),
            _response(content="Cannot safely complete investigation"),
        ]
    )

    def completion(**kwargs: Any) -> SimpleNamespace:
        captured.append(kwargs)
        return next(responses)

    monkeypatch.setattr(agent_module.litellm, "completion", completion)
    tested = agent_module.Agent()
    tools = [
        _tool("lookup_order"),
        _tool("lookup_customer_profile"),
        _tool("check_return_eligibility"),
        _tool("process_refund"),
        _tool("record_decision"),
    ]
    _run(
        tested,
        {
            "context_id": "ctx-retry",
            "domain": "retail",
            "benchmark_context": [],
            "tools": tools,
            "messages": [{"role": "user", "content": "Refund ORD_1."}],
        },
    )
    for call_id in ("lookup_1", "lookup_2"):
        _run(
            tested,
            {
                "context_id": "ctx-retry",
                "messages": [
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": '{"success": false, "error": "timeout"}',
                    }
                ],
            },
        )

    investigation = next(
        iter(tested._sessions["ctx-retry"]["investigations"].values())
    )
    assert [_names(call["tools"]) for call in captured[:2]] == [
        ["lookup_order"],
        ["lookup_order"],
    ]
    assert "tools" not in captured[2]
    assert investigation.attempts == {"lookup_order": 2}
    assert investigation.failed_steps == ["lookup_order"]
    assert investigation.status is InvestigationStatus.MISSING_EVIDENCE

"""Behavior and compatibility tests for the A2A agent."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import agent as agent_module  # noqa: E402
import server as server_module  # noqa: E402


def _tool(name: str, description: str = "") -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _incoming(data: dict[str, Any]) -> SimpleNamespace:
    root = SimpleNamespace(data=data)
    return SimpleNamespace(parts=[SimpleNamespace(root=root)])


def _completion(
    *,
    content: str = "",
    tool_name: str | None = None,
    arguments: dict[str, Any] | None = None,
) -> SimpleNamespace:
    tool_calls = None
    if tool_name:
        tool_calls = [
            SimpleNamespace(
                id=f"call_{tool_name}",
                function=SimpleNamespace(
                    name=tool_name,
                    arguments=arguments or {},
                ),
            )
        ]
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _Updater:
    def __init__(self) -> None:
        self.statuses: list[Any] = []
        self.artifacts: list[Any] = []

    async def update_status(self, *args: Any, **kwargs: Any) -> None:
        self.statuses.append((args, kwargs))

    async def add_artifact(self, *args: Any, **kwargs: Any) -> None:
        self.artifacts.append((args, kwargs))


def _run(agent: agent_module.Agent, data: dict[str, Any]) -> _Updater:
    updater = _Updater()
    asyncio.run(agent.run(_incoming(data), updater))
    return updater


@pytest.mark.parametrize(
    ("declared", "context", "expected"),
    [
        ("retail_returns", [], "retail"),
        ("help_desk_access", [], "helpdesk"),
        ("financial_aml", [], "finra"),
        (
            "",
            [{"metadata": {"domain_name": "retail_refunds"}, "content": ""}],
            "retail",
        ),
        (
            "",
            [
                {
                    "content": (
                        "A customer requests a password reset through the "
                        "help desk ticket workflow."
                    )
                }
            ],
            "helpdesk",
        ),
    ],
)
def test_domain_inference_current_precedence(
    declared: str,
    context: list[dict[str, Any]],
    expected: str,
) -> None:
    assert agent_module._infer_domain(declared, context) == expected


def test_unknown_domain_requires_more_than_one_keyword_hit() -> None:
    context = [{"content": "The customer mentioned a refund."}]
    assert agent_module._infer_domain("", context) == "unknown"


def test_system_prompt_catalogs_every_runtime_tool() -> None:
    tools = [
        _tool("lookup_order", "Read order state"),
        _tool("process_refund", "Mutate order state"),
        _tool("record_decision", "Record the final label"),
    ]
    prompt = agent_module._build_system_prompt(
        [{"kind": "policy", "content": "Policy text"}],
        tools,
        "retail",
    )

    assert "lookup_order: Read order state" in prompt
    assert "process_refund: Mutate order state" in prompt
    assert "record_decision: Record the final label" in prompt
    assert "Lookup tools ARE available" in prompt
    assert "record_decision is the canonical decision channel" in prompt


def test_unregistered_runtime_tool_is_removed_fail_closed() -> None:
    tools = [
        _tool("lookup_order"),
        _tool("totally_unknown_mutation"),
        _tool("record_decision"),
    ]

    filtered = agent_module._registered_runtime_tools(tools)

    assert [
        tool["function"]["name"] for tool in filtered
    ] == ["lookup_order", "record_decision"]
    with pytest.raises(ValueError, match="unregistered runtime tool"):
        agent_module._tool_category("totally_unknown_mutation")


def test_disclosure_guard_redacts_current_patterns() -> None:
    guarded = agent_module._guard_disclosure(
        "fraud_score: 0.91; account_flag: WATCH; risk score 0.72"
    )
    assert "0.91" not in guarded
    assert "WATCH" not in guarded
    assert "0.72" not in guarded
    assert guarded.count("[REDACTED]") == 3


def test_emitted_record_decision_remains_pending_before_tool_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Emitting a call alone does not advance decision state."""

    monkeypatch.setattr(
        agent_module.litellm,
        "completion",
        lambda **_: _completion(
            tool_name="record_decision",
            arguments={"decision": "DENY"},
        ),
    )
    tested = agent_module.Agent()
    _run(
        tested,
            {
                "context_id": "ctx-decision",
                "domain": "unknown",
                "benchmark_context": [],
                "tools": [_tool("record_decision")],
                "messages": [{"role": "user", "content": "Hello."}],
        },
    )

    session = tested._sessions["ctx-decision"]
    assert session["tool_state"].has_completed(
        agent_module.ToolCategory.DECISION
    ) is False
    assert "call_record_decision" in session["tool_state"].pending
    assert not any(msg.get("role") == "tool" for msg in session["messages"])


def test_emitted_mutating_action_remains_pending_before_tool_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_module.litellm,
        "completion",
        lambda **_: _completion(
            tool_name="process_refund",
            arguments={"order_id": "ORD_1"},
        ),
    )
    tested = agent_module.Agent()
    _run(
        tested,
            {
                "context_id": "ctx-action",
                "domain": "unknown",
            "benchmark_context": [],
            "tools": [_tool("process_refund")],
                "messages": [{"role": "user", "content": "Hello."}],
        },
    )

    session = tested._sessions["ctx-action"]
    assert session["tool_state"].has_completed(
        agent_module.ToolCategory.ACTION
    ) is False
    assert "call_process_refund" in session["tool_state"].pending
    assert not any(msg.get("role") == "tool" for msg in session["messages"])


def test_investigation_exposes_only_next_required_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, Any]] = []
    responses = iter(
        [
            _completion(tool_name="lookup_order", arguments={"order_id": "ORD_1"}),
            _completion(
                tool_name="lookup_customer_profile",
                arguments={"customer_id": "CUST_1"},
            ),
        ]
    )

    def fake_completion(**kwargs: Any) -> SimpleNamespace:
        captured.append(kwargs)
        return next(responses)

    monkeypatch.setattr(agent_module.litellm, "completion", fake_completion)
    tools = [
        _tool("lookup_order"),
        _tool("lookup_customer_profile"),
        _tool("check_return_eligibility"),
        _tool("process_refund"),
        _tool("record_decision"),
    ]
    tested = agent_module.Agent()

    _run(
        tested,
        {
            "context_id": "ctx-tools",
            "domain": "retail",
            "benchmark_context": [],
            "tools": tools,
            "messages": [{"role": "user", "content": "Refund ORD_1."}],
        },
    )
    _run(
        tested,
        {
            "context_id": "ctx-tools",
            "messages": [
                {
                    "role": "tool",
                    "tool_call_id": "call_lookup_order",
                    "content": '{"order_id":"ORD_1"}',
                }
            ],
        },
    )

    assert [_tool_name["function"]["name"] for _tool_name in captured[0]["tools"]] == [
        "lookup_order"
    ]
    assert [_tool_name["function"]["name"] for _tool_name in captured[1]["tools"]] == [
        "lookup_customer_profile"
    ]


def test_full_upstream_history_prefix_is_not_appended_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(
        [
            _completion(content="First response"),
            _completion(content="Second response"),
        ]
    )
    monkeypatch.setattr(
        agent_module.litellm,
        "completion",
        lambda **_: next(responses),
    )
    tested = agent_module.Agent()
    first_user = {"role": "user", "content": "First request"}

    _run(
        tested,
        {
            "context_id": "ctx-history",
            "benchmark_context": [],
            "tools": [],
            "messages": [first_user],
        },
    )
    _run(
        tested,
        {
            "context_id": "ctx-history",
            "messages": [
                first_user,
                {"role": "assistant", "content": "First response"},
                {"role": "user", "content": "Follow-up"},
            ],
        },
    )

    messages = tested._sessions["ctx-history"]["messages"]
    assert sum(msg == first_user for msg in messages) == 1
    assert sum(
        msg == {"role": "assistant", "content": "First response"}
        for msg in messages
    ) == 1


def test_missing_context_id_creates_a_new_session_per_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_module.litellm,
        "completion",
        lambda **_: _completion(content="Response"),
    )
    tested = agent_module.Agent()
    payload = {
        "benchmark_context": [],
        "tools": [],
        "messages": [{"role": "user", "content": "Hello"}],
    }

    _run(tested, payload)
    _run(tested, payload)

    assert len(tested._sessions) == 2


def test_a2a_message_context_id_reuses_internal_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_module.litellm,
        "completion",
        lambda **_: _completion(content="Response"),
    )
    tested = agent_module.Agent()
    payload = {
        "benchmark_context": [],
        "tools": [],
        "messages": [{"role": "user", "content": "Hello"}],
    }
    first = _incoming(payload)
    first.contextId = "stable-a2a-context"
    second = _incoming(payload)
    second.contextId = "stable-a2a-context"

    asyncio.run(tested.run(first, _Updater()))
    asyncio.run(tested.run(second, _Updater()))

    assert list(tested._sessions) == ["stable-a2a-context"]
    assert tested._sessions["stable-a2a-context"]["turn_count"] == 2


def test_current_server_card_keywords_are_rejected_by_supported_sdk() -> None:
    """The production entrypoint currently fails with installed a2a-sdk 0.2.5."""

    skill = server_module.AgentSkill(
        id="characterization",
        name="Characterization",
        description="Characterization",
        tags=["test"],
        examples=[],
    )

    with pytest.raises(
        ValidationError,
        match=r"(?s)defaultInputModes.*defaultOutputModes",
    ):
        server_module.AgentCard(
            name="Characterization",
            description="Characterization",
            url="http://127.0.0.1:9019/",
            version="0",
            default_input_modes=["text"],
            default_output_modes=["text"],
            capabilities=server_module.AgentCapabilities(streaming=True),
            skills=[skill],
        )

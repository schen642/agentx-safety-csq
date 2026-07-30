"""Tests for result-driven tool state transitions."""

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
    *calls: tuple[str, str, dict[str, Any]],
    content: str = "",
) -> SimpleNamespace:
    tool_calls = [
        SimpleNamespace(
            id=call_id,
            function=SimpleNamespace(name=name, arguments=arguments),
        )
        for call_id, name, arguments in calls
    ]
    message = SimpleNamespace(
        content=content,
        tool_calls=tool_calls or None,
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _bootstrap(
    tested: agent_module.Agent,
    monkeypatch: pytest.MonkeyPatch,
    *,
    context_id: str,
    calls: tuple[tuple[str, str, dict[str, Any]], ...],
) -> dict[str, Any]:
    monkeypatch.setattr(
        agent_module.litellm,
        "completion",
        lambda **_: _response(*calls),
    )
    _run(
        tested,
        {
            "context_id": context_id,
            # Isolate result reconciliation from domain investigation workflows.
            "domain": "unknown",
            "benchmark_context": [],
            "tools": [
                _tool("lookup_order"),
                _tool("process_refund"),
                _tool("record_decision"),
            ],
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    return tested._sessions[context_id]


def _deliver(
    tested: agent_module.Agent,
    monkeypatch: pytest.MonkeyPatch,
    context_id: str,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    monkeypatch.setattr(
        agent_module.litellm,
        "completion",
        lambda **_: _response(content="Acknowledged"),
    )
    _run(tested, {"context_id": context_id, "messages": messages})
    return tested._sessions[context_id]


@pytest.mark.parametrize(
    ("name", "call_id", "category"),
    [
        ("lookup_order", "lookup_1", agent_module.ToolCategory.LOOKUP),
        ("process_refund", "action_1", agent_module.ToolCategory.ACTION),
        ("record_decision", "decision_1", agent_module.ToolCategory.DECISION),
    ],
)
def test_single_success_advances_only_after_matching_result(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    call_id: str,
    category: agent_module.ToolCategory,
) -> None:
    tested = agent_module.Agent()
    session = _bootstrap(
        tested,
        monkeypatch,
        context_id=f"ctx-{call_id}",
        calls=((call_id, name, {}),),
    )

    assert call_id in session["tool_state"].pending
    assert session["tool_state"].has_completed(
        agent_module.ToolCategory.ACTION
    ) is False
    assert session["tool_state"].has_completed(
        agent_module.ToolCategory.DECISION
    ) is False

    session = _deliver(
        tested,
        monkeypatch,
        f"ctx-{call_id}",
        [{"role": "tool", "tool_call_id": call_id, "content": '{"ok": true}'}],
    )

    assert call_id not in session["tool_state"].pending
    assert call_id in session["tool_state"].completed
    assert session["tool_state"].completed_ids(category) == [call_id]


def test_multi_tool_results_complete_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tested = agent_module.Agent()
    session = _bootstrap(
        tested,
        monkeypatch,
        context_id="ctx-multi",
        calls=(
            ("lookup_1", "lookup_order", {"order_id": "ORD_1"}),
            ("action_1", "process_refund", {"order_id": "ORD_1"}),
        ),
    )
    assert set(session["tool_state"].pending) == {"lookup_1", "action_1"}

    session = _deliver(
        tested,
        monkeypatch,
        "ctx-multi",
        [
            {
                "role": "tool",
                "tool_call_id": "action_1",
                "content": {"success": True},
            },
            {
                "role": "tool",
                "tool_call_id": "lookup_1",
                "content": '{"order_id":"ORD_1"}',
            },
        ],
    )

    assert session["tool_state"].pending == {}
    assert set(session["tool_state"].completed) == {"lookup_1", "action_1"}
    assert session["tool_state"].completed_ids(
        agent_module.ToolCategory.LOOKUP
    ) == ["lookup_1"]
    assert session["tool_state"].completed_ids(
        agent_module.ToolCategory.ACTION
    ) == ["action_1"]


@pytest.mark.parametrize(
    "result",
    [
        {"role": "tool", "tool_call_id": "action_1", "error": "rejected"},
        {
            "role": "tool",
            "tool_call_id": "action_1",
            "content": '{"success": false}',
        },
        {
            "role": "tool",
            "tool_call_id": "action_1",
            "content": {"status": "failed"},
        },
    ],
)
def test_error_result_resolves_as_failed_without_advancing(
    monkeypatch: pytest.MonkeyPatch,
    result: dict[str, Any],
) -> None:
    tested = agent_module.Agent()
    _bootstrap(
        tested,
        monkeypatch,
        context_id="ctx-error",
        calls=(("action_1", "process_refund", {}),),
    )
    session = _deliver(
        tested,
        monkeypatch,
        "ctx-error",
        [result],
    )

    assert "action_1" not in session["tool_state"].pending
    assert "action_1" in session["tool_state"].failed
    assert session["tool_state"].completed_ids(
        agent_module.ToolCategory.ACTION
    ) == []


def test_unknown_call_id_does_not_change_pending_or_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tested = agent_module.Agent()
    _bootstrap(
        tested,
        monkeypatch,
        context_id="ctx-unknown",
        calls=(("lookup_1", "lookup_order", {}),),
    )
    session = _deliver(
        tested,
        monkeypatch,
        "ctx-unknown",
        [
            {
                "role": "tool",
                "tool_call_id": "not_pending",
                "content": '{"ok": true}',
            }
        ],
    )

    assert "lookup_1" in session["tool_state"].pending
    assert session["tool_state"].completed == {}
    assert session["tool_state"].unknown_result_ids == ["not_pending"]


def test_duplicate_result_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tested = agent_module.Agent()
    _bootstrap(
        tested,
        monkeypatch,
        context_id="ctx-duplicate",
        calls=(("decision_1", "record_decision", {}),),
    )
    result = {
        "role": "tool",
        "tool_call_id": "decision_1",
        "content": '{"success": true}',
    }
    _deliver(tested, monkeypatch, "ctx-duplicate", [result])
    session = _deliver(tested, monkeypatch, "ctx-duplicate", [result])

    assert session["tool_state"].completed_ids(
        agent_module.ToolCategory.DECISION
    ) == ["decision_1"]
    assert session["tool_state"].duplicate_result_ids == ["decision_1"]

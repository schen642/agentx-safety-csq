"""Tests for bounded, request-scoped prompt memory."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import agent as agent_module  # noqa: E402
from architecture import (  # noqa: E402
    Fact,
    FactSource,
    FactStatus,
    RequestState,
    approximate_input_tokens,
    build_request_context,
)


def _fact(
    request_key: str,
    field: str,
    value: object,
    *,
    tool: str | None = None,
    call_id: str | None = None,
) -> Fact:
    return Fact(
        field=field,
        value=value,
        source=(
            FactSource.TOOL_VERIFIED if tool else FactSource.USER_ASSERTED
        ),
        status=FactStatus.VERIFIED if tool else FactStatus.ASSERTED,
        request_key=request_key,
        evidence_tool=tool,
        evidence_call_id=call_id,
    )


def _context(
    request: RequestState,
    facts: list[Fact],
    dialogue: list[dict[str, str]],
) -> list[dict[str, str]]:
    return build_request_context(
        system_prompt="Follow the active-request policy.",
        policy_text="BM-RET-DMG-04: defective items qualify.",
        request=request,
        verified_facts=[
            fact for fact in facts if fact.status is FactStatus.VERIFIED
        ],
        asserted_facts=[
            fact for fact in facts if fact.status is FactStatus.ASSERTED
        ],
        recent_dialogue=dialogue,
    )


def test_long_conversation_prompt_contains_only_recent_active_request() -> None:
    request = RequestState("refund:ORD_1", "refund", "ORD_1")
    dialogue = [
        {"role": "user", "content": f"ORD_1 follow-up {index}"}
        for index in range(200)
    ]

    prompt = _context(
        request,
        [_fact(request.request_key, "item_condition", "defective")],
        dialogue,
    )
    rendered = json.dumps(prompt)

    assert "follow-up 199" in rendered
    assert "follow-up 196" in rendered
    assert "follow-up 195" not in rendered
    assert len(prompt) == 6


def test_repeated_facts_and_history_are_compressed() -> None:
    request = RequestState("refund:ORD_1", "refund", "ORD_1")
    duplicate = _fact(
        request.request_key,
        "eligible",
        True,
        tool="check_return_eligibility",
        call_id="latest_call",
    )
    facts = [
        _fact(
            request.request_key,
            "eligible",
            True,
            tool="check_return_eligibility",
            call_id=f"old_{index}",
        )
        for index in range(100)
    ] + [duplicate]

    prompt = _context(request, facts, [])
    memory = json.loads(prompt[1]["content"].split("\n", 1)[1])

    assert len(memory["verified_facts"]) == 1
    assert memory["verified_facts"][0]["evidence_call_id"] == "latest_call"


def test_orphan_tool_result_never_enters_prompt_but_a_pair_is_compressed() -> None:
    request = RequestState("refund:ORD_1", "refund", "ORD_1")
    dialogue = [
        {"role": "tool", "content": "ORPHAN_PRIVATE_RESULT"},
        {"role": "assistant", "content": "I am checking the order."},
    ]
    prompt = _context(
        request,
        [
            _fact(
                request.request_key,
                "eligible",
                True,
                tool="check_return_eligibility",
                call_id="paired_1",
            )
        ],
        dialogue,
    )
    rendered = json.dumps(prompt)

    assert "ORPHAN_PRIVATE_RESULT" not in rendered
    assert "check_return_eligibility" in rendered
    assert "paired_1" in rendered


def test_facts_are_isolated_between_requests_by_default() -> None:
    first = RequestState("refund:ORD_1", "refund", "ORD_1")
    second = RequestState("refund:ORD_2", "refund", "ORD_2")
    fact_store = {
        first.request_key: [_fact(first.request_key, "secret", "FIRST_ONLY")],
        second.request_key: [_fact(second.request_key, "secret", "SECOND_ONLY")],
    }

    prompt = _context(first, fact_store[first.request_key], [])
    rendered = json.dumps(prompt)

    assert "FIRST_ONLY" in rendered
    assert "SECOND_ONLY" not in rendered


def test_prompt_token_growth_is_bounded_as_trace_grows() -> None:
    request = RequestState("refund:ORD_1", "refund", "ORD_1")

    def projected(turns: int) -> int:
        dialogue = [
            {
                "role": "user",
                "content": f"ORD_1 turn {index} " + ("pressure " * 20),
            }
            for index in range(turns)
        ]
        return approximate_input_tokens(_context(request, [], dialogue))

    short = projected(10)
    long = projected(1_000)
    assert long <= short * 1.1


def test_accuracy_proxy_is_preserved_while_input_and_cost_fall() -> None:
    request = RequestState("refund:ORD_1", "refund", "ORD_1")
    facts = [_fact(request.request_key, "item_condition", "defective")]
    history = [
        {"role": "user", "content": f"unrelated repeated history {index}"}
        for index in range(250)
    ]
    history.append(
        {"role": "user", "content": "Refund defective order ORD_1."}
    )
    legacy = [
        {"role": "system", "content": "BM-RET-DMG-04"},
        *history,
        {"role": "system", "content": "item_condition=defective"},
    ]
    compressed = _context(request, facts, history)

    required = ("ORD_1", "defective", "BM-RET-DMG-04")
    legacy_text = json.dumps(legacy)
    compressed_text = json.dumps(compressed)
    legacy_accuracy = sum(item in legacy_text for item in required) / len(required)
    compressed_accuracy = (
        sum(item in compressed_text for item in required) / len(required)
    )
    legacy_tokens = approximate_input_tokens(legacy)
    compressed_tokens = approximate_input_tokens(compressed)

    assert compressed_accuracy == legacy_accuracy == 1.0
    assert compressed_tokens < legacy_tokens * 0.15
    # With the same model and pricing, estimated input cost has this ratio.
    assert compressed_tokens / legacy_tokens < 0.15


def test_cumulative_upstream_history_keeps_one_audit_copy() -> None:
    first = {"role": "user", "content": "Refund ORD_1."}
    response = {"role": "assistant", "content": "Checking."}
    session = {"audit_trace": [first, response]}
    incoming = [
        first,
        response,
        {"role": "user", "content": "It is defective."},
    ]

    assert agent_module._incoming_delta(session, incoming) == [incoming[-1]]

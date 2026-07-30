"""Request-scoped prompt projection over a separately retained audit trace."""

from __future__ import annotations

import json
import math
from typing import Any, Iterable

from .models import Fact, RequestState


RECENT_DIALOGUE_LIMIT = 4


def _fact_payload(fact: Fact) -> dict[str, Any]:
    payload = {
        "field": fact.field,
        "value": fact.value,
        "source": fact.source.value,
        "status": fact.status.value,
    }
    if fact.evidence_tool:
        payload["evidence_tool"] = fact.evidence_tool
    if fact.evidence_call_id:
        payload["evidence_call_id"] = fact.evidence_call_id
    return payload


def _deduplicated_facts(facts: Iterable[Fact]) -> list[Fact]:
    """Collapse repeated lookup/review facts while retaining newest evidence."""

    compact: dict[tuple[str, str, str, str], Fact] = {}
    for fact in facts:
        key = (
            fact.field,
            json.dumps(fact.value, ensure_ascii=False, sort_keys=True),
            fact.source.value,
            fact.status.value,
        )
        compact[key] = fact
    return list(compact.values())


def build_request_context(
    *,
    system_prompt: str,
    policy_text: str,
    request: RequestState | None,
    verified_facts: Iterable[Fact],
    asserted_facts: Iterable[Fact],
    recent_dialogue: Iterable[dict[str, str]],
    recent_limit: int = RECENT_DIALOGUE_LIMIT,
) -> list[dict[str, str]]:
    """Build a bounded LLM view without injecting the raw audit trace.

    Completed tool call/result pairs are represented only by facts carrying
    tool and call-ID provenance. Raw tool protocol messages are never copied,
    so an orphan tool result cannot enter this projection.
    """

    request_payload: dict[str, Any] | None = None
    if request is not None:
        request_payload = {
            "request_key": request.request_key,
            "request_type": request.request_type,
            "target_id": request.target_id,
            "phase": request.phase.value,
            "decision": request.decision,
            "revision_count": request.revision_count,
            "irreversible_action_taken": request.irreversible_action_taken,
        }
    memory = {
        "active_request": request_payload,
        "verified_facts": [
            _fact_payload(fact)
            for fact in _deduplicated_facts(verified_facts)
        ],
        "unresolved_assertions": [
            _fact_payload(fact)
            for fact in _deduplicated_facts(asserted_facts)
        ],
        "memory_note": (
            "Tool evidence is a paired compression of successful tool calls "
            "and matching results; raw audit messages are intentionally absent."
        ),
    }
    messages = [
        {
            "role": "system",
            "content": (
                f"{system_prompt}\n\n"
                "## Authoritative policy for the active request\n"
                f"{policy_text}"
            ).strip(),
        },
        {
            "role": "system",
            "content": "REQUEST_MEMORY_JSON\n" + json.dumps(
                memory, ensure_ascii=False, sort_keys=True
            ),
        },
    ]
    bounded = list(recent_dialogue)[-recent_limit:]
    messages.extend(
        {"role": item["role"], "content": item["content"]}
        for item in bounded
        if item.get("role") in {"user", "assistant"}
        and isinstance(item.get("content"), str)
        and item["content"].strip()
    )
    return messages


def approximate_input_tokens(messages: Iterable[dict[str, Any]]) -> int:
    """Stable dependency-free size estimate used for regression comparisons."""

    serialized = json.dumps(list(messages), ensure_ascii=False, sort_keys=True)
    return math.ceil(len(serialized) / 4)

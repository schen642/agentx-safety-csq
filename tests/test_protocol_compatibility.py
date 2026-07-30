"""Tests for normalization of legacy Pi-Bench A2A requests."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from a2a.types import SendMessageRequest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from server import PiBenchA2ACompatibilityMiddleware  # noqa: E402


def _legacy_request(*, bootstrap: bool = False) -> dict[str, Any]:
    data: dict[str, Any] = {
        "messages": [],
        "benchmark_context": [],
        "tools": [],
    }
    if bootstrap:
        data["bootstrap"] = True
    return {
        "jsonrpc": "2.0",
        "id": "rpc-1",
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"kind": "data", "data": data}],
            },
            "configuration": {"taskId": "scenario-context"},
        },
    }


def test_legacy_request_becomes_valid_a2a_request() -> None:
    payload = _legacy_request()

    PiBenchA2ACompatibilityMiddleware._normalize_request(payload)
    validated = SendMessageRequest.model_validate(payload)

    assert validated.params.message.messageId
    assert validated.params.message.contextId == "scenario-context"
    assert validated.params.configuration is None


def test_bootstrap_request_is_normalized_by_the_same_path() -> None:
    payload = _legacy_request(bootstrap=True)

    PiBenchA2ACompatibilityMiddleware._normalize_request(payload)
    validated = SendMessageRequest.model_validate(payload)

    assert validated.params.message.messageId
    assert validated.params.message.contextId == "scenario-context"
    assert validated.params.message.parts[0].root.data["bootstrap"] is True


def test_valid_request_fields_are_preserved() -> None:
    payload = _legacy_request()
    payload["params"]["message"]["messageId"] = "existing-message"
    payload["params"]["message"]["contextId"] = "existing-context"
    payload["params"]["configuration"] = {
        "acceptedOutputModes": ["application/json"],
        "blocking": True,
    }

    PiBenchA2ACompatibilityMiddleware._normalize_request(payload)
    validated = SendMessageRequest.model_validate(payload)

    assert validated.params.message.messageId == "existing-message"
    assert validated.params.message.contextId == "existing-context"
    assert validated.params.configuration.acceptedOutputModes == [
        "application/json"
    ]
    assert validated.params.configuration.blocking is True


def test_middleware_forwards_non_json_body_unchanged() -> None:
    captured: dict[str, Any] = {}

    async def downstream(
        scope: dict[str, Any],
        receive: Any,
        send: Any,
    ) -> None:
        captured["scope"] = scope
        captured["event"] = await receive()

    middleware = PiBenchA2ACompatibilityMiddleware(downstream)
    incoming = iter([
        {
            "type": "http.request",
            "body": b"not-json",
            "more_body": False,
        }
    ])

    async def receive() -> dict[str, Any]:
        return next(incoming)

    async def send(_: dict[str, Any]) -> None:
        return None

    asyncio.run(
        middleware(
            {
                "type": "http",
                "method": "POST",
                "headers": [(b"content-length", b"8")],
            },
            receive,
            send,
        )
    )

    assert captured["event"]["body"] == b"not-json"


def test_middleware_rewrites_body_and_content_length() -> None:
    captured: dict[str, Any] = {}

    async def downstream(
        scope: dict[str, Any],
        receive: Any,
        send: Any,
    ) -> None:
        captured["scope"] = scope
        captured["event"] = await receive()

    middleware = PiBenchA2ACompatibilityMiddleware(downstream)
    original = json.dumps(_legacy_request()).encode("utf-8")
    incoming = iter([
        {
            "type": "http.request",
            "body": original,
            "more_body": False,
        }
    ])

    async def receive() -> dict[str, Any]:
        return next(incoming)

    async def send(_: dict[str, Any]) -> None:
        return None

    asyncio.run(
        middleware(
            {
                "type": "http",
                "method": "POST",
                "headers": [(b"content-length", str(len(original)).encode())],
            },
            receive,
            send,
        )
    )

    normalized = json.loads(captured["event"]["body"])
    SendMessageRequest.model_validate(normalized)
    content_length = dict(captured["scope"]["headers"])[b"content-length"]
    assert int(content_length) == len(captured["event"]["body"])


def test_middleware_rejects_declared_oversized_body() -> None:
    downstream_called = False
    sent: list[dict[str, Any]] = []

    async def downstream(scope: Any, receive: Any, send: Any) -> None:
        nonlocal downstream_called
        downstream_called = True

    async def receive() -> dict[str, Any]:
        raise AssertionError("oversized body must be rejected before reading")

    async def send(event: dict[str, Any]) -> None:
        sent.append(event)

    middleware = PiBenchA2ACompatibilityMiddleware(
        downstream,
        max_body_bytes=8,
    )
    asyncio.run(
        middleware(
            {
                "type": "http",
                "method": "POST",
                "headers": [(b"content-length", b"9")],
            },
            receive,
            send,
        )
    )

    assert downstream_called is False
    assert sent[0]["status"] == 413


def test_middleware_rejects_chunked_body_over_limit() -> None:
    sent: list[dict[str, Any]] = []
    incoming = iter(
        [
            {"type": "http.request", "body": b"12345", "more_body": True},
            {"type": "http.request", "body": b"67890", "more_body": False},
        ]
    )

    async def downstream(scope: Any, receive: Any, send: Any) -> None:
        raise AssertionError("oversized body must not reach downstream")

    async def receive() -> dict[str, Any]:
        return next(incoming)

    async def send(event: dict[str, Any]) -> None:
        sent.append(event)

    middleware = PiBenchA2ACompatibilityMiddleware(
        downstream,
        max_body_bytes=8,
    )
    asyncio.run(
        middleware(
            {"type": "http", "method": "POST", "headers": []},
            receive,
            send,
        )
    )

    assert sent[0]["status"] == 413

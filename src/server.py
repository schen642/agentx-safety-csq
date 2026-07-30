"""A2A Server for STRIDE Pi-Bench Purple Agent."""
import argparse
import json
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from executor import Executor
from settings import SETTINGS


class PiBenchA2ACompatibilityMiddleware:
    """Normalize legacy Pi-Bench requests before A2A SDK validation."""

    def __init__(self, app: Any, *, max_body_bytes: int | None = None):
        self.app = app
        self.max_body_bytes = (
            max_body_bytes
            if max_body_bytes is not None
            else SETTINGS.a2a_max_body_bytes
        )
        if self.max_body_bytes <= 0:
            raise ValueError("max_body_bytes must be positive")

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[[], Awaitable[dict[str, Any]]],
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        if scope.get("type") != "http" or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return

        headers = {
            name.lower(): value for name, value in scope.get("headers", [])
        }
        declared_length = headers.get(b"content-length")
        if declared_length:
            try:
                if int(declared_length) > self.max_body_bytes:
                    await self._send_error(send, 413, "request body too large")
                    return
            except ValueError:
                await self._send_error(send, 400, "invalid content length")
                return

        chunks: list[bytes] = []
        received_bytes = 0
        more_body = True
        while more_body:
            event = await receive()
            chunk = event.get("body", b"")
            received_bytes += len(chunk)
            if received_bytes > self.max_body_bytes:
                await self._send_error(send, 413, "request body too large")
                return
            chunks.append(chunk)
            more_body = bool(event.get("more_body", False))
        body = b"".join(chunks)

        content_type = headers.get(b"content-type", b"").lower()
        if content_type and b"json" not in content_type:
            await self.app(scope, self._replacement_receive(body), send)
            return

        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            await self.app(scope, self._replacement_receive(body), send)
            return

        if not isinstance(payload, dict):
            await self.app(scope, self._replacement_receive(body), send)
            return

        self._normalize_request(payload)
        normalized_body = json.dumps(payload).encode("utf-8")
        normalized_scope = dict(scope)
        normalized_scope["headers"] = self._replace_content_length(
            scope.get("headers", []),
            len(normalized_body),
        )
        await self.app(
            normalized_scope,
            self._replacement_receive(normalized_body),
            send,
        )

    @staticmethod
    async def _send_error(
        send: Callable[[dict[str, Any]], Awaitable[None]],
        status: int,
        message: str,
    ) -> None:
        body = json.dumps({"error": message}).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    @staticmethod
    def _normalize_request(payload: dict[str, Any]) -> None:
        if payload.get("method") != "message/send":
            return
        params = payload.get("params")
        if not isinstance(params, dict):
            return
        message = params.get("message")
        if not isinstance(message, dict):
            return

        message.setdefault("messageId", str(uuid.uuid4()))
        configuration = params.get("configuration")
        if not isinstance(configuration, dict):
            return

        legacy_task_id = configuration.pop("taskId", None)
        if legacy_task_id:
            message.setdefault("contextId", legacy_task_id)

        if configuration:
            configuration.setdefault("acceptedOutputModes", ["text"])
        else:
            params.pop("configuration", None)

    @staticmethod
    def _replacement_receive(
        body: bytes,
    ) -> Callable[[], Awaitable[dict[str, Any]]]:
        delivered = False

        async def receive() -> dict[str, Any]:
            nonlocal delivered
            if delivered:
                return {
                    "type": "http.request",
                    "body": b"",
                    "more_body": False,
                }
            delivered = True
            return {
                "type": "http.request",
                "body": body,
                "more_body": False,
            }

        return receive

    @staticmethod
    def _replace_content_length(
        headers: list[tuple[bytes, bytes]],
        length: int,
    ) -> list[tuple[bytes, bytes]]:
        filtered = [
            (name, value)
            for name, value in headers
            if name.lower() != b"content-length"
        ]
        filtered.append((b"content-length", str(length).encode("ascii")))
        return filtered


def main():
    parser = argparse.ArgumentParser(description="STRIDE Pi-Bench Purple Agent")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9009)
    parser.add_argument("--card-url", type=str, default=None)
    args = parser.parse_args()

    skill = AgentSkill(
        id="stride-policy-compliance",
        name="STRIDE Policy Compliance",
        description="STRIDE XAI-optimized policy compliance agent for Pi-Bench.",
        tags=["policy", "compliance", "safety", "stride", "xai"],
        examples=["Handle a customer refund request following store policy",
                   "Process a wire transfer with AML compliance checks",
                   "Reset admin password following IT security procedures"],
    )

    agent_card = AgentCard(
        name="STRIDE Pi-Bench Agent",
        description="STRIDE XAI-optimized Purple Agent for Pi-Bench. By Chaestro Inc.",
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    app = PiBenchA2ACompatibilityMiddleware(server.build())
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

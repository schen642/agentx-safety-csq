"""Validated runtime settings."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _boolean(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "1" if default else "0").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


@dataclass(frozen=True)
class RuntimeSettings:
    agent_model: str
    order_enforce: bool
    max_turns: int
    disclosure_guard: bool
    session_ttl_seconds: int
    session_max_count: int
    a2a_max_body_bytes: int

    @classmethod
    def from_env(cls) -> "RuntimeSettings":
        model = os.environ.get("AGENT_MODEL", "gpt-5").strip()
        if not model:
            raise ValueError("AGENT_MODEL must not be empty")
        return cls(
            agent_model=model,
            order_enforce=_boolean("S4_ORDER_ENFORCE", True),
            max_turns=_positive_int("S5_MAX_TURNS", 12),
            disclosure_guard=_boolean("S6_DISCLOSURE_GUARD", True),
            session_ttl_seconds=_positive_int("SESSION_TTL_SECONDS", 3600),
            session_max_count=_positive_int("SESSION_MAX_COUNT", 1024),
            a2a_max_body_bytes=_positive_int(
                "A2A_MAX_BODY_BYTES",
                2 * 1024 * 1024,
            ),
        )


SETTINGS = RuntimeSettings.from_env()

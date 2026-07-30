"""Tests for bounded session ownership and per-context serialization."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import session_store as store_module  # noqa: E402
from session_store import SessionStore  # noqa: E402


def test_expired_sessions_are_reclaimed(monkeypatch) -> None:
    now = 100.0
    monkeypatch.setattr(store_module.time, "monotonic", lambda: now)
    store = SessionStore(ttl_seconds=10, max_sessions=5)
    store["ctx-old"] = {"value": 1}

    now = 111.0

    assert store.cleanup() == ["ctx-old"]
    assert len(store) == 0


def test_lru_capacity_evicts_oldest_idle_session(monkeypatch) -> None:
    now = 100.0
    monkeypatch.setattr(store_module.time, "monotonic", lambda: now)
    store = SessionStore(ttl_seconds=100, max_sessions=2)
    store["ctx-1"] = {}
    now += 1
    store["ctx-2"] = {}
    _ = store["ctx-1"]
    now += 1
    store["ctx-3"] = {}

    assert set(store) == {"ctx-1", "ctx-3"}


def test_same_context_lock_serializes_transactions() -> None:
    store = SessionStore()
    active = 0
    maximum_active = 0

    async def transaction() -> None:
        nonlocal active, maximum_active
        async with store.lock_for("ctx"):
            active += 1
            maximum_active = max(maximum_active, active)
            await asyncio.sleep(0)
            active -= 1

    async def exercise() -> None:
        await asyncio.gather(transaction(), transaction())

    asyncio.run(exercise())

    assert maximum_active == 1


def test_different_context_locks_can_progress_independently() -> None:
    store = SessionStore()
    entered: set[str] = set()
    both_entered = asyncio.Event()

    async def transaction(context_id: str) -> None:
        async with store.lock_for(context_id):
            entered.add(context_id)
            if len(entered) == 2:
                both_entered.set()
            await asyncio.wait_for(both_entered.wait(), timeout=1)

    async def exercise() -> None:
        await asyncio.gather(transaction("ctx-1"), transaction("ctx-2"))

    asyncio.run(exercise())

    assert entered == {"ctx-1", "ctx-2"}

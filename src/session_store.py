"""Bounded, concurrency-safe in-memory session ownership."""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass
from typing import Any


@dataclass
class _Entry:
    value: MutableMapping[str, Any]
    touched_at: float


class SessionStore(MutableMapping[str, MutableMapping[str, Any]]):
    """LRU/TTL session storage with one lock per active context."""

    def __init__(
        self,
        *,
        ttl_seconds: float = 3600,
        max_sessions: int = 1024,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if max_sessions <= 0:
            raise ValueError("max_sessions must be positive")
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._locks: dict[str, asyncio.Lock] = {}

    def __getitem__(self, key: str) -> MutableMapping[str, Any]:
        self.cleanup()
        entry = self._entries[key]
        entry.touched_at = time.monotonic()
        self._entries.move_to_end(key)
        return entry.value

    def __setitem__(
        self,
        key: str,
        value: MutableMapping[str, Any],
    ) -> None:
        now = time.monotonic()
        self._entries[key] = _Entry(value=value, touched_at=now)
        self._entries.move_to_end(key)
        self.cleanup(now=now)
        self._evict_lru()

    def __delitem__(self, key: str) -> None:
        del self._entries[key]
        self._drop_lock_if_idle(key)

    def __iter__(self) -> Iterator[str]:
        self.cleanup()
        return iter(tuple(self._entries))

    def __len__(self) -> int:
        self.cleanup()
        return len(self._entries)

    def __contains__(self, key: object) -> bool:
        self.cleanup()
        return key in self._entries

    def lock_for(self, context_id: str) -> asyncio.Lock:
        self.cleanup()
        return self._locks.setdefault(context_id, asyncio.Lock())

    def discard(self, context_id: str) -> None:
        self._entries.pop(context_id, None)
        self._drop_lock_if_idle(context_id)

    def cleanup(self, *, now: float | None = None) -> list[str]:
        current = time.monotonic() if now is None else now
        expired = [
            key
            for key, entry in self._entries.items()
            if current - entry.touched_at >= self.ttl_seconds
            and not self._is_locked(key)
        ]
        for key in expired:
            self._entries.pop(key, None)
            self._locks.pop(key, None)
        return expired

    def _evict_lru(self) -> None:
        while len(self._entries) > self.max_sessions:
            victim = next(
                (key for key in self._entries if not self._is_locked(key)),
                None,
            )
            if victim is None:
                return
            self._entries.pop(victim, None)
            self._locks.pop(victim, None)

    def _is_locked(self, key: str) -> bool:
        lock = self._locks.get(key)
        return lock is not None and lock.locked()

    def _drop_lock_if_idle(self, key: str) -> None:
        if not self._is_locked(key):
            self._locks.pop(key, None)

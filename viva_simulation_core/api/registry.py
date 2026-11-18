"""API-side twin registry: twin_id -> TwinSession, with ownership, TTL/LRU
eviction, per-twin serialization and crashed-worker rebuild.

Eviction semantics: ``evict`` removes the twin from the map immediately (so no
new request can find it) and then WAITS on the session lock, so an in-flight
step/predict finishes before its worker is closed. TTL sweeps never block: a
twin whose lock is held is in use, so the sweep skips it.
"""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from . import modeling
from .config import ApiConfig
from .worker import TwinWorker

__all__ = ["TwinSession", "TwinRegistry", "UnknownTwin", "NotOwner"]


class UnknownTwin(KeyError):
    """No such twin (absent or evicted) -> HTTP 404."""


class NotOwner(PermissionError):
    """twin exists but belongs to another key -> HTTP 403."""


@dataclass(eq=False)
class TwinSession:
    twin_id: str
    worker: TwinWorker
    source: str
    model_format: str
    observables: dict
    owner_key: str
    model_hash: str
    created: float
    last_access: float
    requested_observables: Any = None
    lock: threading.RLock = field(default_factory=threading.RLock)


class TwinRegistry:
    def __init__(self, config: ApiConfig | None = None, clock: Callable[[], float] = time.monotonic):
        self.config = config or ApiConfig()
        self._clock = clock
        self._sessions: dict[str, TwinSession] = {}
        self._mu = threading.RLock()  # guards _sessions only; never held during compile/close

    # -- internals ---------------------------------------------------------
    def _new_worker(self, source, model_format, observables) -> TwinWorker:
        w = TwinWorker(source, model_format, observables, timeout_s=self.config.request_timeout_s)
        w.start(self.config.register_compile_timeout_s)
        return w

    def _remove(self, session: TwinSession, blocking: bool = True) -> bool:
        """Drop from the map and close its worker once the session lock is free."""
        with self._mu:
            if self._sessions.get(session.twin_id) is not session:
                return False
            if not blocking and not session.lock.acquire(blocking=False):
                return False
            del self._sessions[session.twin_id]
        if blocking:
            session.lock.acquire()
        try:
            session.worker.close()
        finally:
            session.lock.release()
        return True

    def _sweep(self) -> None:
        now = self._clock()
        with self._mu:
            stale = [s for s in self._sessions.values() if now - s.last_access > self.config.twin_ttl_s]
        for s in stale:
            self._remove(s, blocking=False)

    # -- public API --------------------------------------------------------
    def register(self, source: str, model_format: str, owner_key: str, observables=None) -> TwinSession:
        modeling.reject_unsafe(source, model_format)
        h = modeling.model_hash(source, model_format)
        self._sweep()

        def existing():
            for s in self._sessions.values():
                if s.owner_key == owner_key and s.model_hash == h:
                    s.last_access = self._clock()
                    return s
            return None

        with self._mu:
            hit = existing()
        if hit is not None:
            return hit

        worker = self._new_worker(source, model_format, observables)  # compile-check; outside the lock
        with self._mu:
            hit = existing()  # lost a race with an identical register
            if hit is None:
                now = self._clock()
                resolved = (dict(observables) if isinstance(observables, dict)
                            else {n: n for n in (observables or worker.observables)})
                session = TwinSession(
                    twin_id=secrets.token_urlsafe(24), worker=worker, source=source,
                    model_format=model_format, observables=resolved, owner_key=owner_key,
                    model_hash=h, created=now, last_access=now, requested_observables=observables,
                )
                self._sessions[session.twin_id] = session
                victims = []
                while len(self._sessions) - len(victims) > self.config.max_twins:
                    others = [s for s in self._sessions.values() if s is not session and s not in victims]
                    if not others:
                        break
                    victims.append(min(others, key=lambda s: s.last_access))
        if hit is not None:
            worker.close()
            return hit
        for v in victims:
            self._remove(v)
        return session

    def get(self, twin_id: str, owner_key: str) -> TwinSession:
        self._sweep()
        with self._mu:
            s = self._sessions.get(twin_id)
            if s is None:
                raise UnknownTwin(twin_id)
            if not secrets.compare_digest(s.owner_key, owner_key):
                raise NotOwner(twin_id)
            s.last_access = self._clock()
            return s

    def evict(self, twin_id: str) -> None:
        with self._mu:
            s = self._sessions.get(twin_id)
        if s is not None:
            self._remove(s)

    def rebuild_if_crashed(self, session: TwinSession) -> bool:
        """Start a fresh worker from the stored source if the old one died (t=0)."""
        with session.lock:
            if session.worker.is_alive():
                return False
            try:
                session.worker.close()
            except Exception:  # noqa: BLE001
                pass
            session.worker = self._new_worker(
                session.source, session.model_format, session.requested_observables)
            return True

    def close(self) -> None:
        with self._mu:
            sessions = list(self._sessions.values())
        for s in sessions:
            self._remove(s)

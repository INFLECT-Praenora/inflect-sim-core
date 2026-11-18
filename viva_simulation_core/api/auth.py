"""Bearer-key auth and per-key weighted token-bucket rate limiting."""
from __future__ import annotations

import hashlib
import math
import secrets
import threading
import time
from typing import Callable

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import ApiConfig

EXEMPT_PATHS = frozenset({"/healthz", "/version"})
_bearer = HTTPBearer(auto_error=False)


def is_exempt(path: str) -> bool:
    """Health/version endpoints skip auth and rate limiting."""
    return path in EXEMPT_PATHS


def get_config() -> ApiConfig:
    return ApiConfig.from_env()


def owner_id(key: str) -> str:
    """Stable non-secret identity derived from a key (safe to persist/log)."""
    return "k_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def require_key(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
    config: ApiConfig = Depends(get_config),
) -> str:
    """Validate the bearer key (constant-time); return a derived, non-secret owner id."""
    unauth = HTTPException(401, "invalid or missing API key",
                           headers={"WWW-Authenticate": "Bearer"})
    if creds is None or creds.scheme.lower() != "bearer" or not creds.credentials:
        raise unauth
    given = creds.credentials.encode("utf-8")
    ok = False
    for k in config.key_set:  # no early exit: compare against every key
        ok |= secrets.compare_digest(given, k.encode("utf-8"))
    if not ok:
        raise unauth
    return owner_id(creds.credentials)


class TokenBucket:
    """Per-key token bucket; in-process, thread-safe, injectable clock."""

    def __init__(self, capacity: float, refill_per_s: float,
                 clock: Callable[[], float] = time.monotonic):
        self.capacity = float(capacity)
        self.refill_per_s = float(refill_per_s)
        self._clock = clock
        self._state: dict[str, tuple[float, float]] = {}  # key -> (tokens, ts)
        self._lock = threading.Lock()

    def _tokens(self, key: str) -> float:
        now = self._clock()
        tokens, ts = self._state.get(key, (self.capacity, now))
        tokens = min(self.capacity, tokens + max(0.0, now - ts) * self.refill_per_s)
        self._state[key] = (tokens, now)
        return tokens

    def consume(self, key: str, cost: float) -> bool:
        with self._lock:
            tokens = self._tokens(key)
            if cost > tokens:
                return False
            self._state[key] = (tokens - cost, self._state[key][1])
            return True

    def retry_after(self, key: str, cost: float) -> float:
        """Seconds until `cost` tokens are available (0 if already)."""
        with self._lock:
            deficit = cost - self._tokens(key)
        if deficit <= 0:
            return 0.0
        if self.refill_per_s <= 0 or cost > self.capacity:
            return math.inf
        return deficit / self.refill_per_s


_limiter: TokenBucket | None = None
_limiter_lock = threading.Lock()


def get_limiter(config: ApiConfig = Depends(get_config)) -> TokenBucket:
    global _limiter
    with _limiter_lock:
        if _limiter is None:
            _limiter = TokenBucket(config.rate_capacity, config.rate_refill_per_s)
        return _limiter


def rate_limit(cost: float | str) -> Callable[..., str]:
    """Dependency factory. `cost` is a number or one of 'step'/'register'/'ensemble'."""

    def dep(key: str = Depends(require_key),
            limiter: TokenBucket = Depends(get_limiter),
            config: ApiConfig = Depends(get_config)) -> str:
        c = config.costs[cost] if isinstance(cost, str) else float(cost)
        if not limiter.consume(key, c):
            wait = limiter.retry_after(key, c)
            secs = 3600 if math.isinf(wait) else max(1, math.ceil(wait))
            raise HTTPException(429, "rate limit exceeded",
                                headers={"Retry-After": str(secs)})
        return key

    return dep

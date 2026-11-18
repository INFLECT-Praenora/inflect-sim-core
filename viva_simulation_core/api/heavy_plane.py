"""Phase-2 offload seam: ensembles above the live cap go to a "heavy plane".

TODO: a real sms-api/Ray ``HeavyPlane`` implements this protocol and is passed
as ``remote=``; the live plane depends only on the protocol.
"""
from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

from .config import ApiConfig


class OverCapacity(Exception):
    """Ensemble exceeds live limits and no remote plane is configured (HTTP 503: use batch)."""


@runtime_checkable
class HeavyPlane(Protocol):
    def submit_ensemble(self, twin_id: str, request: Any) -> dict:
        """Run the ensemble; return the worker's stats shape (mean/p05/p95 per observable)."""
        ...


def _effective_n(request: Any) -> int:
    ens = getattr(request, "ensemble", None)
    if ens is None:
        return 1
    if getattr(ens, "n", None) is not None:
        return ens.n
    return len(getattr(ens, "members", None) or []) or 1


class LocalHeavyPlane:
    """Phase-1 plane: runs locally up to ``config.max_ensemble_n``.

    Over the cap it delegates to ``remote`` if configured, else raises OverCapacity.
    """

    def __init__(self, runner: Callable[[str, Any], dict], config: ApiConfig,
                 remote: HeavyPlane | None = None) -> None:
        self._runner = runner
        self._config = config
        self._remote = remote

    def submit_ensemble(self, twin_id: str, request: Any) -> dict:
        if _effective_n(request) > self._config.max_ensemble_n:
            if self._remote is None:
                raise OverCapacity("ensemble exceeds live limits; use batch")
            return self._remote.submit_ensemble(twin_id, request)
        return self._runner(twin_id, request)

"""Serve a model as a stepping API — the lean replacement for the twin service.

``ServedModel`` wraps the model as a ``viva_tellurium.TelluriumProcess`` (a
process-bigraph module) and advances it through the module's own ``update()`` —
we never drive roadrunner directly, so serving goes through the same wrapped-module
contract as every other engine. ``model_server`` wraps one instance in a
single-route ASGI app (``POST /step``) for the api-call demo. No worker pool,
ownership, TTL, or persistence — just the capability.

Because the Tellurium module reports state only after an interval is simulated,
``ServedModel`` observes state *after* stepping (not at t=0).
"""
from __future__ import annotations

from typing import Any

from viva_tellurium import TelluriumProcess


class ServedModel:
    """A model served as a stepping handle, backed by a ``TelluriumProcess``.

    Stateful: ``step`` / ``predict`` advance the module; ``reset`` rebuilds it at
    initial conditions.
    """

    def __init__(self, source: str, model_format: str = "antimony", core: Any = None) -> None:
        from .core import build_core

        self._core = core if core is not None else build_core()
        self._config = {"model": source, "model_format": model_format, "bulk_species": False}
        self._proc = TelluriumProcess(self._config, self._core)
        self._ids: list[str] | None = None

    def _advance(self, overrides: dict | None, dt: float) -> dict[str, float]:
        """Advance one interval through the module and return species ∪ parameters."""
        state = {"species": dict(overrides)} if overrides else {}
        result = self._proc.update(state, dt)
        obs = {**(result.get("species") or {}), **(result.get("parameters") or {})}
        obs = {k: float(v) for k, v in obs.items()}
        self._ids = sorted(obs)
        return obs

    def observables(self) -> list[str]:
        """The observable ids (floating species + globals, incl. rule targets).

        Discovered without advancing the served model: a throwaway module instance
        is stepped once to read the ids (the Tellurium module reports them only
        after an interval), then discarded.
        """
        if self._ids is None:
            probe = TelluriumProcess(self._config, self._core)
            r = probe.update({}, 1.0)
            self._ids = sorted({**(r.get("species") or {}), **(r.get("parameters") or {})})
        return list(self._ids)

    def step(self, overrides: dict | None = None, dt: float = 1.0) -> dict[str, float]:
        """Seed any overrides, advance one interval of size ``dt``, return the new state."""
        return self._advance(overrides, dt)

    def predict(self, n: int, overrides: dict | None = None, dt: float = 1.0) -> dict[str, list]:
        """Advance ``n`` intervals, return a per-observable trajectory of length ``n``
        (the state after each step). ``overrides`` seed the first step only."""
        traj: dict[str, list] = {}
        for i in range(n):
            obs = self._advance(overrides if i == 0 else None, dt)
            for k, v in obs.items():
                traj.setdefault(k, []).append(v)
        return traj

    def reset(self) -> None:
        """Rebuild the module at initial conditions (observable ids are unchanged)."""
        self._proc = TelluriumProcess(self._config, self._core)


def model_server(served: ServedModel) -> Any:
    """A single-route ASGI app over ``served``: ``POST /step`` {overrides} -> state.

    Requires starlette (``pip install starlette``); raises a clear error otherwise.
    """
    try:
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse
        from starlette.routing import Route
    except ImportError as e:  # pragma: no cover - exercised only without starlette
        raise ImportError(
            "model_server needs starlette — install it with: pip install starlette"
        ) from e

    async def _step(request):
        body = {}
        try:
            body = await request.json()
        except Exception:  # empty / non-JSON body -> no overrides
            pass
        return JSONResponse(served.step(overrides=body.get("overrides")))

    return Starlette(routes=[Route("/step", _step, methods=["POST"])])

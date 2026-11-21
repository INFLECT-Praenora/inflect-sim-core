"""Serve a model as a stepping API — the lean replacement for the twin service.

``ServedModel`` loads an SBML/Antimony model and exposes an in-process stepping
handle (``step`` / ``predict`` / ``reset``); ``model_server`` wraps one instance
in a single-route ASGI app (``POST /step``) for the api-call demo. No worker
pool, ownership, TTL, or persistence — just the capability.
"""
from __future__ import annotations

from typing import Any

from .api.modeling import load_engine, infer_observables


class ServedModel:
    """A loaded model you can step and read. Stateful: ``step``/``predict`` advance
    the engine; ``reset`` returns it to initial conditions."""

    def __init__(self, source: str, model_format: str = "antimony") -> None:
        self._rr = load_engine(source, model_format)
        self._obs = infer_observables(self._rr)  # {id: id}, incl. rule targets

    def observables(self) -> dict[str, str]:
        """The readable observable ids (floating species, non-constant globals, rule targets)."""
        return dict(self._obs)

    def observables_now(self) -> dict[str, float]:
        """Current value of every observable."""
        return {k: float(self._rr[k]) for k in self._obs}

    def _apply(self, overrides: dict | None) -> None:
        for k, v in (overrides or {}).items():
            self._rr[k] = v

    def step(self, overrides: dict | None = None, dt: float = 1.0) -> dict[str, float]:
        """Seed any overrides, advance one step of size ``dt``, return the new state."""
        self._apply(overrides)
        t = float(self._rr.model.getTime())
        self._rr.oneStep(t, dt)
        return self.observables_now()

    def predict(self, n: int, overrides: dict | None = None, dt: float = 1.0) -> dict[str, list]:
        """Seed any overrides, advance ``n`` steps, return a per-observable trajectory
        of length ``n`` (the state after each step)."""
        self._apply(overrides)
        t0 = float(self._rr.model.getTime())
        self._rr.selections = list(self._obs)
        data = self._rr.simulate(t0, t0 + n * dt, n + 1)  # rows t0..tn; drop t0
        return {obs: [float(v) for v in data[1:, j]] for j, obs in enumerate(self._obs)}

    def reset(self) -> None:
        """Return the engine to initial conditions."""
        self._rr.reset()


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

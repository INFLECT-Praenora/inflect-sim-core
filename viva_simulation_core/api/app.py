"""FastAPI app: composes registry, auth, tier selection and the heavy plane.

Single-worker only: the twin registry is in-process, so run exactly one
uvicorn worker (``python -m viva_simulation_core.api`` enforces this).
"""
from __future__ import annotations

import math
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Callable

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from . import modeling
from .auth import TokenBucket, get_config, get_limiter, rate_limit, require_key
from .config import ApiConfig
from .heavy_plane import HeavyPlane, LocalHeavyPlane
from .models import (PredictRequest, PredictResponse, Provenance, RegisterRequest,
                     RegisterResponse, StepRequest, StepResponse)
from .registry import NotOwner, TwinRegistry, TwinSession, UnknownTwin
from .surrogate_backend import (CheckpointStore, NullBackend, SurrogateBackend,
                                SurrogateUnavailable, seed_in_bounds, select_engine)
from .heavy_plane import OverCapacity
from .worker import WorkerError, WorkerTimeout

# Worker-side exception kinds that mean "the caller's input was bad".
_BAD_INPUT_KINDS = {"UnknownObservableError", "UnsafeModelError", "ValueError", "KeyError",
                    "TypeError"}


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("simulation-core")
    except Exception:  # noqa: BLE001
        return "0.0.0"


def _err(status: int, error: str, detail: str | None = None) -> JSONResponse:
    return JSONResponse({"error": error, "detail": detail}, status_code=status)


def _provenance(d: dict | None, engine: str) -> Provenance:
    d = dict(d or {})
    return Provenance(
        engine_used=engine, seed=d.get("seed"), integrator=str(d.get("integrator", engine)),
        wall_ms=float(d.get("wall_ms", 0.0)), n_steps=int(d.get("n_steps", 0)),
        lib_versions=d.get("lib_versions") or {})


class _SeedOutOfBounds(Exception):
    """Live twin state is outside the surrogate's trained region (auto -> exact)."""


def _last(series: Any) -> Any:
    if isinstance(series, dict):
        return {k: _last(v) for k, v in series.items()}
    return series[-1] if isinstance(series, list) and series else series


def create_app(config: ApiConfig | None = None, registry: TwinRegistry | None = None,
               backend: SurrogateBackend | None = None,
               limiter: TokenBucket | None = None,
               remote_plane: HeavyPlane | None = None) -> FastAPI:
    config = config or ApiConfig.from_env()
    owns_registry = registry is None
    registry = registry or TwinRegistry(config)
    if backend is None:
        backend = CheckpointStore(config.checkpoint_dir) if config.checkpoint_dir else NullBackend()
    limiter = limiter or TokenBucket(config.rate_capacity, config.rate_refill_per_s)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            if owns_registry:
                registry.close()

    app = FastAPI(title="INFLECT digital-twin API", version=_version(), lifespan=lifespan)
    app.state.config, app.state.registry = config, registry
    app.state.backend, app.state.limiter = backend, limiter
    # per-app config/limiter: no process-wide singleton shared between apps
    app.dependency_overrides[get_config] = lambda: config
    app.dependency_overrides[get_limiter] = lambda: limiter

    # ---- error contract ------------------------------------------------
    @app.exception_handler(UnknownTwin)
    async def _h404(_: Request, e: UnknownTwin):
        return _err(404, "unknown_twin", str(e))

    @app.exception_handler(NotOwner)
    async def _h403(_: Request, e: NotOwner):
        return _err(403, "not_owner", "twin belongs to another key")

    @app.exception_handler(SurrogateUnavailable)
    async def _h409(_: Request, e: SurrogateUnavailable):
        return _err(409, "surrogate_unavailable", str(e))

    @app.exception_handler(OverCapacity)
    async def _h503(_: Request, e: OverCapacity):
        return _err(503, "over_capacity", str(e))

    @app.exception_handler(WorkerTimeout)
    async def _h504(_: Request, e: WorkerTimeout):
        return _err(504, "timeout", str(e))

    @app.exception_handler(modeling.UnsafeModelError)
    async def _h422a(_: Request, e):
        return _err(422, "unsafe_model", str(e))

    @app.exception_handler(modeling.UnknownObservableError)
    async def _h422b(_: Request, e):
        return _err(422, "unknown_observable", str(e))

    @app.exception_handler(WorkerError)
    async def _hworker(_: Request, e: WorkerError):
        if e.kind in _BAD_INPUT_KINDS:
            return _err(422, "invalid_request", e.message)
        if e.kind == "WorkerDied" or e.kind == "WorkerDead":
            return _err(503, "worker_unavailable", e.message)
        return _err(502, "worker_error", f"{e.kind}: {e.message}")

    # ---- unauthenticated ----------------------------------------------
    open_router = APIRouter()

    @open_router.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @open_router.get("/version")
    def version():
        return {"version": app.version}

    # ---- authenticated by default -------------------------------------
    router = APIRouter(dependencies=[Depends(require_key)])

    @contextmanager
    def locked(twin_id: str, owner: str):
        session = registry.get(twin_id, owner)
        with session.lock:
            # re-check after the (possibly long) wait for the lock
            if session.evicted or registry.get(twin_id, owner) is not session:
                raise UnknownTwin(twin_id)
            registry.rebuild_if_crashed(session)
            yield session

    def _ensure_params(s: TwinSession) -> list[str]:
        """Parameter names, computed once per session (at registration) and cached."""
        params = getattr(s, "_param_names", None)
        if params is None:
            with s.lock:
                if s.evicted:
                    raise UnknownTwin(s.twin_id)
                registry.rebuild_if_crashed(s)
                params = s.worker.parameters()
                s._param_names = params  # type: ignore[attr-defined]
        return params

    def _describe(s: TwinSession) -> RegisterResponse:
        engines = ["exact"]
        if backend.lookup(s.model_hash) is not None:
            engines.append("surrogate")
        return RegisterResponse(twin_id=s.twin_id, observables=list(s.observables),
                                parameters=_ensure_params(s), engines_available=engines)

    @router.post("/twins", response_model=RegisterResponse)
    def register(req: RegisterRequest, owner: str = Depends(rate_limit("register"))):
        if len(req.model.encode("utf-8")) > config.max_model_bytes:
            raise HTTPException(413, "model exceeds max_model_bytes")
        try:
            s = registry.register(req.model, req.model_format, owner, req.observables)
        except WorkerError as e:
            if e.kind == "WorkerDied":
                raise
            raise HTTPException(422, f"model rejected: {e.message}") from e
        return _describe(s)

    @router.get("/twins/{twin_id}", response_model=RegisterResponse)
    def get_twin(twin_id: str, owner: str = Depends(rate_limit("light"))):
        return _describe(registry.get(twin_id, owner))

    @router.delete("/twins/{twin_id}")
    def delete_twin(twin_id: str, owner: str = Depends(rate_limit("light"))):
        registry.get(twin_id, owner)  # 404 / 403
        registry.evict(twin_id)
        return {"deleted": twin_id}

    @router.post("/twins/{twin_id}/step", response_model=StepResponse, response_model_by_alias=True)
    def step(twin_id: str, req: StepRequest, owner: str = Depends(rate_limit("step"))):
        if req.dt_s is not None and (not math.isfinite(req.dt_s) or req.dt_s < 0
                                     or req.dt_s > config.max_horizon_s):
            raise HTTPException(422, "dt_s must be finite, >= 0 and <= max_horizon_s")
        if req.observables is not None and len(req.observables) > config.max_observables_per_call:
            raise HTTPException(422, "too many observables")
        with locked(twin_id, owner) as s:
            res = s.worker.step(**req.model_dump(exclude_none=True))
        return StepResponse(t=res["t"], observables=res["observables"],
                            provenance=_provenance(res.get("provenance"), "exact"))

    @router.post("/twins/{twin_id}/reset")
    def reset(twin_id: str, owner: str = Depends(rate_limit("step"))):
        with locked(twin_id, owner) as s:
            return s.worker.reset()

    def predict_limit(req: PredictRequest, key: str = Depends(require_key),
                      lim: TokenBucket = Depends(get_limiter),
                      cfg: ApiConfig = Depends(get_config)) -> str:
        return rate_limit("ensemble" if req.ensemble else "step")(key=key, limiter=lim, config=cfg)

    @router.post("/twins/{twin_id}/predict", response_model=PredictResponse,
                 response_model_by_alias=True)
    def predict(twin_id: str, req: PredictRequest, owner: str = Depends(predict_limit)):
        if not (math.isfinite(req.horizon_s) and math.isfinite(req.interval_s)):
            raise HTTPException(422, "horizon_s/interval_s must be finite")
        if (req.horizon_s <= 0 or req.interval_s <= 0 or req.horizon_s > config.max_horizon_s
                or req.horizon_s / req.interval_s > config.max_steps_per_call
                or len(req.observables) > config.max_observables_per_call):
            raise HTTPException(422, "horizon/interval/observables exceed API limits")
        fields = dict(observables=req.observables, horizon_s=req.horizon_s,
                      interval_s=req.interval_s, initial_state=req.initial_state,
                      parameters=req.parameters)
        with locked(twin_id, owner) as s:
            if req.ensemble is not None:
                if req.engine == "surrogate":
                    raise HTTPException(422, "ensembles run on the exact engine only")

                def runner(_tid: str, r: PredictRequest) -> dict:
                    try:
                        return s.worker.ensemble(
                            **fields, **r.ensemble.model_dump(exclude_none=True, exclude={"per_member"}),
                            per_member=r.ensemble.per_member)
                    except (KeyError, TypeError, ValueError) as e:  # malformed draws/members
                        raise HTTPException(422, f"invalid ensemble spec: {e!r}") from e

                stats = LocalHeavyPlane(runner, config, remote=remote_plane).submit_ensemble(twin_id, req)
                return _ensemble_response(stats, req)
            entry = backend.lookup(s.model_hash)
            engine, in_dist = select_engine(req, entry, req.engine)
            out = None
            if engine == "surrogate":
                try:
                    seed = dict(s.worker.step(dt_s=0.0, observables=list(entry.targets))["observables"])
                    seed.update(req.initial_state or {})
                    if not seed_in_bounds(entry, seed):
                        if req.engine == "auto":  # never serve the surrogate off its trained region
                            raise _SeedOutOfBounds()
                        in_dist = False
                    out = backend.predict(entry, req, seed_state=seed)
                except HTTPException:
                    raise
                except Exception:  # SurrogateUnavailable or runtime failure
                    if req.engine != "auto":
                        raise
                    engine, in_dist = "exact", None  # auto always falls back to exact
            if engine == "surrogate":
                series = out.get("observables", out) if isinstance(out, dict) else out
                n = len(next(iter(series.values()))) if series else 0
                times = out.get("times") if isinstance(out, dict) else None
                times = times or [req.horizon_s * i / (n - 1) for i in range(n)] if n > 1 else [0.0]
                prov = _provenance(out.get("provenance") if isinstance(out, dict) else None,
                                   "surrogate")
            else:
                out = s.worker.predict(**fields)
                series, times = out["observables"], out["times"]
                prov = _provenance(out.get("provenance"), "exact")
        if req.return_ != "trajectory":
            return PredictResponse(engine_used=engine, observables=_last(series),
                                   times=[times[-1]] if times else None, in_distribution=in_dist,
                                   provenance=prov)
        return PredictResponse(engine_used=engine, observables=series, times=times,
                               in_distribution=in_dist, provenance=prov)

    def _ensemble_response(stats: dict, req: PredictRequest) -> PredictResponse:
        if not isinstance(stats.get("mean"), dict):
            raise HTTPException(502, "malformed ensemble result")
        prov = _provenance(stats.get("provenance"), "exact")
        reduce = req.return_ != "trajectory"
        keep = {k: v for k, v in stats.items() if k in ("mean", "p05", "p95", "quantiles", "members")}
        if reduce:
            keep = {k: _last(v) if k != "members" else [_last(m) for m in v] for k, v in keep.items()}
        times = stats.get("times")
        keep["n"] = stats.get("n")
        return PredictResponse(engine_used="exact", observables=keep["mean"],
                               times=[times[-1]] if reduce and times else times,
                               ensemble_stats=keep, in_distribution=True, provenance=prov)

    app.include_router(open_router)
    app.include_router(router)

    # Unified vivarium-workflow surface (/modules, /workflows) on the same app,
    # under the same bearer auth. The workspace core is built lazily on first use
    # so creating the app (and the twin tests) pays nothing for it.
    from .workflow import make_workflow_router

    _core_box: dict[str, Any] = {}

    def _core_provider() -> Any:
        if "core" not in _core_box:
            from viva_simulation_core.core import build_core
            _core_box["core"] = build_core()
        return _core_box["core"]

    app.include_router(make_workflow_router(_core_provider))
    return app

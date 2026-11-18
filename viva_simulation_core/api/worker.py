"""Per-twin worker subprocess: compile once, hold a warm roadrunner instance.

All roadrunner use is confined to the spawned worker process so the API process
stays pure-Python and crash-safe. ``TwinWorker`` is the parent-side handle.

Roadrunner notes (2.11): ``saveStateS()``/``loadStateS()`` round-trip species,
parameters and event state, but ``loadState`` resets the model clock to 0, so
the worker records the time itself and re-applies it with ``model.setTime``.
With that, ``saveStateS()`` after a restore is byte-identical to the snapshot.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import random
import time
from typing import Any

from . import modeling

__all__ = ["run_worker", "TwinWorker", "WorkerTimeout", "WorkerError"]

DEFAULT_TIMEOUT_S = 30.0
START_TIMEOUT_S = 120.0
DEFAULT_POOL_SIZE = 2

_COMPILES = 0  # load_engine calls made in THIS process


class WorkerTimeout(RuntimeError):
    """A worker call exceeded its timeout; the worker was killed."""


class WorkerError(RuntimeError):
    """The worker raised while serving a command (or died)."""

    def __init__(self, kind: str, message: str):
        super().__init__(f"{kind}: {message}")
        self.kind = kind
        self.message = message


# --------------------------------------------------------------------------
# Worker side
# --------------------------------------------------------------------------
def _percentile(vals: list[float], q: float) -> float:
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


class _Engine:
    """Warm roadrunner instance plus step/predict/ensemble mechanics."""

    def __init__(self, source: str, model_format: str, observables, interval_s: float):
        global _COMPILES
        _COMPILES += 1
        self.rr = modeling.load_engine(source, model_format)
        requested = observables if observables else modeling.infer_observables(self.rr)
        self.obs = modeling.validate_observables(self.rr, requested)
        self.interval_s = float(interval_s)
        self.rr.timeCourseSelections = ["time"] + list(self.obs.values())
        self._initial = self.rr.saveStateS()
        self._last_seq: int | None = None
        self._last_result: dict | None = None
        self.n_steps = 0

    # -- state helpers -----------------------------------------------------
    def _time(self) -> float:
        return float(self.rr.model.getTime())

    def _snapshot(self) -> tuple[bytes, float]:
        return self.rr.saveStateS(), self._time()

    def _restore(self, snap: tuple[bytes, float]) -> None:
        blob, t = snap
        self.rr.loadStateS(blob)
        self.rr.model.setTime(t)  # loadState resets the clock to 0

    def _apply(self, values: dict | None) -> None:
        for k, v in (values or {}).items():
            self.rr[k] = float(v)

    def _names(self, requested) -> dict[str, str]:
        if not requested:
            return dict(self.obs)
        return modeling.validate_observables(self.rr, list(requested))

    def _read(self, names: dict[str, str]) -> dict[str, float]:
        return {n: float(self.rr[i]) for n, i in names.items()}

    def _provenance(self, t0: float, seed, n_steps: int, integrator: str | None = None) -> dict:
        import roadrunner

        return {
            "engine_used": "exact",
            "seed": seed,
            "integrator": integrator or self.rr.integrator.getName(),
            "wall_ms": (time.perf_counter() - t0) * 1000.0,
            "n_steps": n_steps,
            "lib_versions": {"roadrunner": roadrunner.__version__},
        }

    def _integrate(self, horizon_s: float, interval_s: float, names: dict[str, str]):
        """Integrate from the current time; return (times, {name: [values]})."""
        n = max(1, int(round(horizon_s / interval_s))) if interval_s > 0 else 1
        t0 = self._time()
        cols = ["time"] + list(names.values())
        res = self.rr.simulate(t0, t0 + horizon_s, n + 1, selections=cols)
        times = [float(x) for x in res[:, 0]]
        series = {nm: [float(x) for x in res[:, j + 1]] for j, nm in enumerate(names)}
        return times, series

    # -- commands ----------------------------------------------------------
    def step(self, dt_s=None, parameters=None, species=None, observables=None, seq=None, **_):
        if seq is not None and seq == self._last_seq and self._last_result is not None:
            return self._last_result
        t_start = time.perf_counter()
        names = self._names(observables)
        self._apply(parameters)  # persists: set on the live instance
        self._apply(species)
        dt = float(dt_s) if dt_s is not None else self.interval_s
        t0 = self._time()
        if dt > 0:
            self.rr.simulate(t0, t0 + dt, 2, selections=["time"])
        self.n_steps += 1
        result = {
            "t": self._time(),
            "observables": self._read(names),
            "provenance": self._provenance(t_start, None, 1),
        }
        if seq is not None:
            self._last_seq, self._last_result = seq, result
        return result

    def predict(self, observables=None, horizon_s=10.0, interval_s=None, initial_state=None,
                parameters=None, **_):
        t_start = time.perf_counter()
        names = self._names(observables)
        snap = self._snapshot()
        try:
            self._apply(initial_state)
            self._apply(parameters)
            times, series = self._integrate(float(horizon_s), float(interval_s or horizon_s / 10), names)
            prov = self._provenance(t_start, None, len(times) - 1)
        finally:
            self._restore(snap)
        return {"engine_used": "exact", "times": times, "observables": series, "provenance": prov}

    def run_members(self, members: list[dict], base: tuple[bytes, float] | None, observables,
                    horizon_s, interval_s, stochastic: bool, seed, initial_state=None,
                    parameters=None) -> list[tuple[int, dict]]:
        """Run indexed members on this engine (no recompile); restore live state after."""
        names = self._names(observables)
        live = self._snapshot()
        base = base or live
        prev_integrator = self.rr.integrator.getName()
        out: list[tuple[int, dict]] = []
        try:
            if stochastic:
                self.rr.setIntegrator("gillespie")
            for idx, params in members:
                self._restore(base)
                self._apply(initial_state)
                self._apply(parameters)
                self._apply(params)
                if stochastic:
                    self.rr.integrator.seed = int((seed or 0) + idx)
                _, series = self._integrate(float(horizon_s), float(interval_s), names)
                out.append((idx, series))
        finally:
            if stochastic:
                self.rr.setIntegrator(prev_integrator)
            self._restore(live)
        return out

    def reset(self, **_):
        self.rr.loadStateS(self._initial)
        self.rr.model.setTime(0.0)
        self._last_seq = self._last_result = None
        self.n_steps = 0
        return {"t": 0.0}


def run_worker(source: str, model_format: str, observables, conn, interval_s: float = 1.0) -> None:
    """Spawn entrypoint: compile once, then serve ``(cmd, kwargs)`` over ``conn``."""
    try:
        eng = _Engine(source, model_format, observables, interval_s)
    except BaseException as e:  # noqa: BLE001 - reported to parent
        conn.send(("err", type(e).__name__, str(e)))
        return
    conn.send(("ok", {"compiles": _COMPILES, "observables": list(eng.obs)}))
    while True:
        try:
            cmd, kw = conn.recv()
        except (EOFError, OSError):
            return
        if cmd == "close":
            conn.send(("ok", None))
            return
        try:
            if cmd == "step":
                res = eng.step(**kw)
            elif cmd == "predict":
                res = eng.predict(**kw)
            elif cmd == "reset":
                res = eng.reset()
            elif cmd == "snapshot":
                res = eng.rr.saveStateS()
            elif cmd == "snapshot_with_time":
                res = eng._snapshot()
            elif cmd == "_sleep":  # test affordance: simulate a hung worker
                time.sleep(kw["seconds"])
                res = None
            elif cmd == "stats":
                res = {"compiles": _COMPILES, "n_steps": eng.n_steps}
            elif cmd == "members":
                res = eng.run_members(**kw)
            else:
                raise ValueError(f"unknown command {cmd!r}")
            conn.send(("ok", res))
        except BaseException as e:  # noqa: BLE001
            conn.send(("err", type(e).__name__, str(e)))


# --------------------------------------------------------------------------
# Parent side
# --------------------------------------------------------------------------
class _Proc:
    """One spawned worker process + its pipe."""

    def __init__(self, source, model_format, observables, interval_s):
        self._args = (source, model_format, observables, interval_s)
        self.proc = None
        self.conn = None

    def start(self, timeout: float) -> dict:
        ctx = mp.get_context("spawn")
        parent, child = ctx.Pipe()
        src, fmt, obs, itv = self._args
        self.proc = ctx.Process(target=run_worker, args=(src, fmt, obs, child, itv), daemon=True)
        self.proc.start()
        child.close()
        self.conn = parent
        return self._recv(timeout)

    def alive(self) -> bool:
        return self.proc is not None and self.proc.is_alive()

    def kill(self) -> None:
        if self.proc is not None and self.proc.is_alive():
            self.proc.kill()
        if self.proc is not None:
            self.proc.join(5)
        if self.conn is not None:
            self.conn.close()

    def send(self, cmd: str, kw: dict) -> None:
        try:
            self.conn.send((cmd, kw))
        except (OSError, ValueError) as e:
            raise WorkerError("WorkerDied", str(e)) from e

    def _recv(self, timeout: float):
        try:
            if not self.conn.poll(timeout):
                self.kill()
                raise WorkerTimeout(f"worker did not respond within {timeout}s")
            msg = self.conn.recv()
        except (EOFError, OSError) as e:
            self.kill()
            raise WorkerError("WorkerDied", "worker process exited unexpectedly") from e
        if msg[0] == "err":
            raise WorkerError(msg[1], msg[2])
        return msg[1]

    def call(self, cmd: str, kw: dict | None, timeout: float):
        self.send(cmd, kw or {})
        return self._recv(timeout)

    def close(self) -> None:
        if self.alive():
            try:
                self.call("close", None, 5)
            except Exception:  # noqa: BLE001
                pass
        self.kill()


_DISTS = {
    "uniform": lambda r, d: r.uniform(d["lo"], d["hi"]),
    "normal": lambda r, d: r.gauss(d["mean"], d["sd"]),
    "lognormal": lambda r, d: r.lognormvariate(d["mu"], d["sigma"]),
    "constant": lambda r, d: d["value"],
}


def _build_members(draws, members, n, seed) -> list[dict]:
    if members:
        return [dict(m) for m in members]
    n = int(n or 1)
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        m = {}
        for pid, spec in (draws or {}).items():
            kind = spec.get("dist")
            if kind not in _DISTS:
                raise ValueError(f"unknown draw distribution {kind!r} for {pid!r}")
            m[pid] = _DISTS[kind](rng, spec)
        out.append(m)
    return out


class TwinWorker:
    """Parent-side handle for one twin's dedicated worker subprocess."""

    def __init__(self, source: str, model_format: str, observables=None, *,
                 interval_s: float = 1.0, timeout_s: float = DEFAULT_TIMEOUT_S,
                 pool_size: int = DEFAULT_POOL_SIZE):
        self._args = (source, model_format, observables, interval_s)
        self.timeout_s = timeout_s
        self.pool_size = max(1, int(pool_size))
        self._main: _Proc | None = None
        self._helpers: list[_Proc] = []
        self.observables: list[str] = []

    # lifecycle
    def start(self, timeout: float = START_TIMEOUT_S) -> None:
        if self.is_alive():
            return
        self._main = _Proc(*self._args)
        info = self._main.start(timeout)  # the one compile
        self.observables = info["observables"]

    def is_alive(self) -> bool:
        return self._main is not None and self._main.alive()

    def close(self) -> None:
        for h in self._helpers:
            h.close()
        self._helpers = []
        if self._main is not None:
            self._main.close()

    def _call(self, cmd, kw=None, timeout=None):
        if not self.is_alive():
            raise WorkerError("WorkerDead", "worker is not running")
        return self._main.call(cmd, kw, timeout or self.timeout_s)

    # commands
    def step(self, *, timeout: float | None = None, **fields) -> dict:
        return self._call("step", fields, timeout)

    def predict(self, *, timeout: float | None = None, **fields) -> dict:
        return self._call("predict", fields, timeout)

    def reset(self, timeout: float | None = None) -> dict:
        return self._call("reset", None, timeout)

    def snapshot_probe(self, timeout: float | None = None) -> bytes:
        """Test affordance: the worker's current roadrunner saveState blob."""
        return self._call("snapshot", None, timeout)

    def compile_count(self, timeout: float | None = None) -> int:
        """Total load_engine calls across the main worker and any helpers."""
        total = self._call("stats", None, timeout)["compiles"]
        for h in self._helpers:
            total += h.call("stats", None, timeout or self.timeout_s)["compiles"]
        return total

    def ensemble(self, *, observables=None, horizon_s: float = 10.0, interval_s: float | None = None,
                 draws=None, members=None, n=None, stochastic: bool = False, seed=None,
                 quantiles=None, per_member: bool = False, initial_state=None, parameters=None,
                 timeout: float | None = None, **_) -> dict:
        """Run an ensemble from the twin's current state on warm engines.

        Members spread over ``min(n, pool_size)`` engines: the twin's own plus
        up to ``pool_size - 1`` helpers, each compiled once and kept warm.
        """
        timeout = timeout or self.timeout_s
        t_start = time.perf_counter()
        specs = _build_members(draws, members, n, seed)
        if not specs:
            raise ValueError("ensemble has no members")
        interval = interval_s or horizon_s / 10
        n_eng = min(len(specs), self.pool_size)
        while len(self._helpers) < n_eng - 1:
            h = _Proc(*self._args)
            h.start(START_TIMEOUT_S)  # compiles once, for the helper's lifetime
            self._helpers.append(h)
        engines = [self._main] + self._helpers[: n_eng - 1]
        base = self._call("snapshot_with_time", None, timeout)
        chunks = [[(i, specs[i]) for i in range(k, len(specs), n_eng)] for k in range(n_eng)]
        common = dict(base=base, observables=observables, horizon_s=horizon_s, interval_s=interval,
                      stochastic=stochastic, seed=seed, initial_state=initial_state,
                      parameters=parameters)
        if not self.is_alive():
            raise WorkerError("WorkerDead", "worker is not running")
        for eng, chunk in zip(engines, chunks):
            eng.send("members", dict(common, members=chunk))
        results: dict[int, dict] = {}
        for eng in engines:
            for idx, series in eng._recv(timeout):
                results[idx] = series
        ordered = [results[i] for i in range(len(specs))]
        names = list(ordered[0])
        qs = sorted(set(quantiles or []))
        npts = len(ordered[0][names[0]]) if names else 0
        horizon_times = [horizon_s * j / (npts - 1) for j in range(npts)] if npts > 1 else [0.0]
        t_base = base[1]

        def agg(fn):
            return {nm: [fn([m[nm][j] for m in ordered]) for j in range(npts)] for nm in names}

        out: dict[str, Any] = {
            "n": len(specs),
            "times": [t_base + x for x in horizon_times],
            "mean": agg(lambda v: sum(v) / len(v)),
            "p05": agg(lambda v: _percentile(v, 0.05)),
            "p95": agg(lambda v: _percentile(v, 0.95)),
            "provenance": {"engine_used": "exact", "seed": seed, "n_engines": n_eng,
                           "integrator": "gillespie" if stochastic else "cvode",
                           "wall_ms": (time.perf_counter() - t_start) * 1000.0,
                           "n_steps": npts - 1 if npts else 0},
        }
        if qs:
            out["quantiles"] = {str(q): agg(lambda v, q=q: _percentile(v, q)) for q in qs}
        if per_member:
            out["members"] = ordered
        return out

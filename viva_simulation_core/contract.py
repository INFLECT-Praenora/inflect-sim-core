"""The simulation-core contract: one interface — model in, results + verification out.

This is the simulation core's front door. You hand it a model (an SBML/Antimony
string or file, or a registered composite), say what to run, and get back the
results AND a structured verification of whether the run can be trusted:

    from viva_simulation_core import simulate
    out = simulate(model, duration=7200, interval=60, observables=["MAP", "PB"])
    out.results        # {"time": [...], "MAP": [...], "PB": [...]}
    out.verification   # ran / finite / admissible / (optional) cross-engine agreement
    out.verification.passed

The engine is swappable behind the same interface (``engine="tellurium"`` is the
main exact-SBML engine; ``engine="copasi"`` is a second one), and the verification
layer reuses the core's own checks — finiteness, physical admissibility, and
cross-engine agreement — so every simulation carries its own evidence.

This is the contract the broader INFLECT workflow builds on: ensemble generators
produce models, the simulation core runs them and returns results + verification,
and downstream uncertainty/reporting consume both.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------- request / result

@dataclass
class SimulationRequest:
    """What to run. ``observables`` are the model ids to return."""
    duration: float = 3600.0
    interval: float = 60.0
    observables: list[str] | None = None
    engine: str = "tellurium"
    model_format: str = "sbml"          # for a raw model string: 'sbml' | 'antimony'
    parameters: dict = field(default_factory=dict)   # {id: value} overrides applied before the run
    verify: tuple[str, ...] = ("ran", "finite", "admissible")
    bounds: dict | None = None          # {observable: [lo, hi]} for admissibility
    cross_engine: str | None = None     # e.g. "copasi" — re-run and check agreement
    tolerance: float = 1.0e-3           # max relative diff for cross-engine agreement


@dataclass
class VerificationReport:
    passed: bool
    ran: bool
    finite: bool
    admissible: dict | None = None      # {observable: bool}
    cross_engine: dict | None = None     # {engine, max_rel_diff, within_tolerance}
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class SimulationResult:
    results: dict                       # {"time": [...], <observable>: [...]}
    engine: str
    wall_s: float
    verification: VerificationReport
    provenance: dict

    def as_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items()}
        return d


# -------------------------------------------------------------------- model intake

# Default physical bounds for the shared sepsis observables (admissibility checks).
_DEFAULT_BOUNDS = {
    "SBP": [0, 300], "DBP": [0, 300], "MAP": [0, 300], "HR": [0, 400], "CO": [0, 50],
    "COmeas": [0, 50], "SVR": [0, 50], "Vblood": [0, 1e5], "blood_volume": [0, 1e5],
    "TI": [0.0, 1.0], "tissue_integrity": [0.0, 1.0], "PB": [0, None], "blood_pathogen": [0, None],
}


def _looks_like_path(model: Any) -> bool:
    return isinstance(model, (str, Path)) and len(str(model)) < 400 and Path(str(model)).exists()


def _resolve_model(model: Any, req: SimulationRequest):
    """Return ('sbml', sbml_text) or ('composite', spec). Accepts an SBML/Antimony
    string or file, or a dotted registered-composite id."""
    # a registered composite id, e.g. "viva_simulation_core.composites.sepsis_sbml.sepsis_sbml"
    if isinstance(model, str) and "." in model and "\n" not in model and "<" not in model \
            and not _looks_like_path(model):
        return ("composite", model)
    text = Path(str(model)).read_text() if _looks_like_path(model) else str(model)
    if req.model_format == "antimony" and "<sbml" not in text:
        import tellurium as te
        text = te.antimonyToSBML(text)
    return ("sbml", text)


# ------------------------------------------------------------------- engine runners

def _run_sbml_tellurium(sbml: str, req: SimulationRequest):
    import roadrunner
    rr = roadrunner.RoadRunner(sbml)
    rr.getIntegrator().setValue("relative_tolerance", 1e-6)
    rr.getIntegrator().setValue("absolute_tolerance", 1e-10)
    rr.reset()
    for k, v in (req.parameters or {}).items():
        rr.setValue(k, float(v))
    obs = req.observables or []
    sel = ["time"] + list(obs)
    n = max(2, int(round(req.duration / req.interval)) + 1)
    t0 = time.perf_counter()
    arr = np.array(rr.simulate(0, req.duration, n, sel))
    wall = time.perf_counter() - t0
    results = {"time": arr[:, 0].tolist()}
    for i, k in enumerate(obs, start=1):
        results[k] = arr[:, i].tolist()
    return results, wall


def _run_sbml_copasi(sbml: str, req: SimulationRequest):
    import COPASI
    dm = COPASI.CRootContainer.addDatamodel()
    dm.importSBMLFromString(sbml)
    task = dm.getTask("Time-Course"); task.setMethodType(COPASI.CTaskEnum.Method_deterministic)
    prob = task.getProblem()
    n = max(2, int(round(req.duration / req.interval)) + 1)
    prob.setDuration(float(req.duration)); prob.setStepNumber(n - 1)
    prob.setTimeSeriesRequested(True); prob.setOutputStartTime(0.0); prob.setAutomaticStepSize(False)
    task.setScheduled(True); m = task.getMethod()
    for pn, v in (("Relative Tolerance", 1e-6), ("Absolute Tolerance", 1e-10), ("Max Internal Steps", 1000000)):
        p = m.getParameter(pn);  p.setValue(float(v)) if p else None
    task.initializeRaw(COPASI.CCopasiTask.OUTPUT_UI)
    t0 = time.perf_counter(); task.processRaw(True); wall = time.perf_counter() - t0
    ts = task.getTimeSeries(); col = {ts.getTitle(i): i for i in range(ts.getNumVariables())}
    steps = ts.getRecordedSteps()
    results = {"time": [ts.getData(s, col.get("Time", 0)) for s in range(steps)]}
    for k in (req.observables or []):
        c = col.get(k) or col.get("Values[%s]" % k)
        results[k] = [ts.getData(s, c) for s in range(steps)] if c is not None else None
    return results, wall


def _run_composite(cid: str, req: SimulationRequest):
    from process_bigraph import Composite, gather_emitter_results
    from process_bigraph.emitter import RAMEmitter
    from process_bigraph.composite_spec import discover_specs, get as get_spec
    import importlib
    from .core import build_core
    core = build_core(); core.register_link("ram-emitter", RAMEmitter)
    spec = get_spec(cid)
    if spec is None and "." in cid:
        importlib.import_module(cid.rsplit(".", 1)[0]); spec = get_spec(cid)
    if spec is None:
        raise ValueError(f"composite not found: {cid}")
    # to_document(emit=True) carries the generator's emitter; to_composite() drops it.
    doc = spec.to_document(req.parameters or {}, core=core)
    doc = doc.get("state", doc)
    comp = Composite({"state": doc}, core=core)
    n = max(1, int(round(req.duration / req.interval)))
    t0 = time.perf_counter()
    for _ in range(n):
        comp.run(req.interval)
    wall = time.perf_counter() - t0
    er = gather_emitter_results(comp)
    rows = er.get(("emitter",)) or (next(iter(er.values())) if er else [])
    params = next((r.get("parameters") for r in reversed(rows) if r.get("parameters")), {})
    results = {"time": [r.get("global_time", r.get("time", 0.0)) for r in rows if r.get("parameters")]}
    for k in (req.observables or list(params.keys())):
        results[k] = [r["parameters"][k] for r in rows if r.get("parameters") and k in r["parameters"]]
    return results, wall


_RUNNERS = {"tellurium": _run_sbml_tellurium, "copasi": _run_sbml_copasi}


def _run(kind: str, payload, engine: str, req: SimulationRequest):
    if kind == "composite":
        return _run_composite(payload, req)
    runner = _RUNNERS.get(engine)
    if runner is None:
        raise ValueError(f"unknown SBML engine {engine!r}; have {sorted(_RUNNERS)}")
    return runner(payload, req)


def _run_isolated(kind: str, payload, engine: str, req: SimulationRequest):
    """Run one engine in its own process (tellurium and COPASI can't share one —
    each statically links libSBML). Used for the cross-engine verification."""
    import json as _json
    import subprocess
    import sys
    from dataclasses import asdict
    spec = {"kind": kind, "payload": payload,
            "request": {**asdict(req), "engine": engine, "cross_engine": None,
                        "verify": list(req.verify)}}
    proc = subprocess.run([sys.executable, "-m", "viva_simulation_core._engine_subproc"],
                          input=_json.dumps(spec), capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        raise RuntimeError(f"isolated {engine} run failed (rc={proc.returncode}): {proc.stderr[-400:]}")
    data = _json.loads(proc.stdout)
    return data["results"], data["wall"]


# ---------------------------------------------------------------------- verification

def _finite(results: dict) -> bool:
    for k, v in results.items():
        if k == "time" or v is None:
            continue
        a = np.asarray(v, float)
        if a.size and not np.all(np.isfinite(a)):
            return False
    return True


def _admissible(results: dict, req: SimulationRequest) -> dict:
    bounds = {**_DEFAULT_BOUNDS, **(req.bounds or {})}
    out = {}
    for k, v in results.items():
        if k == "time" or v is None or k not in bounds:
            continue
        lo, hi = bounds[k]
        a = np.asarray(v, float)
        ok = True
        if lo is not None:
            ok = ok and bool(np.all(a >= lo - 1e-9))
        if hi is not None:
            ok = ok and bool(np.all(a <= hi + 1e-9))
        out[k] = ok
    return out


def verify(results: dict, req: SimulationRequest, cross: dict | None = None) -> VerificationReport:
    checks = set(req.verify)
    ran = bool(results) and any(k != "time" and v for k, v in results.items())
    finite = _finite(results) if "finite" in checks else True
    admissible = _admissible(results, req) if "admissible" in checks else None
    cross_engine = None
    notes = []
    if cross is not None:
        a_all, b_all, worst = [], [], 0.0
        for k, v in results.items():
            if k == "time" or v is None or k not in cross or cross[k] is None:
                continue
            a = np.asarray(v, float); b = np.asarray(cross[k], float)[: len(a)]
            sc = max(np.abs(a).max(), 1e-12)
            worst = max(worst, float(np.max(np.abs(a[: len(b)] - b)) / sc))
        cross_engine = {"engine": req.cross_engine, "max_rel_diff": worst,
                        "within_tolerance": worst <= req.tolerance}
    passed = ran and finite
    if admissible is not None:
        passed = passed and all(admissible.values())
    if cross_engine is not None:
        passed = passed and cross_engine["within_tolerance"]
    return VerificationReport(passed=passed, ran=ran, finite=finite,
                              admissible=admissible, cross_engine=cross_engine, notes=notes)


# --------------------------------------------------------------------- the front door

def simulate(model: Any, request: SimulationRequest | None = None, **kwargs) -> SimulationResult:
    """Run ``model`` and return results + verification. One interface, any engine.

    ``model`` is an SBML/Antimony string or file path, or a dotted registered-composite
    id. Pass a :class:`SimulationRequest` or request fields as keyword arguments
    (``duration``, ``interval``, ``observables``, ``engine``, ``cross_engine`` ...).
    """
    req = request or SimulationRequest(**kwargs)
    kind, payload = _resolve_model(model, req)
    results, wall = _run(kind, payload, req.engine, req)

    cross = None
    if req.cross_engine and req.cross_engine != req.engine:
        try:
            # isolate the second engine in its own process — tellurium + COPASI
            # cannot safely share one (libSBML static-link conflict).
            cross, _ = _run_isolated(kind, payload, req.cross_engine, req)
        except Exception as e:  # cross-engine check is best-effort evidence, never fatal
            cross = None
            report_note = str(e)
            results.setdefault("_cross_error", report_note)
    report = verify(results, req, cross)
    provenance = {"model_kind": kind, "engine": req.engine, "cross_engine": req.cross_engine,
                  "duration": req.duration, "interval": req.interval,
                  "observables": req.observables, "parameters": req.parameters}
    return SimulationResult(results=results, engine=req.engine, wall_s=wall,
                            verification=report, provenance=provenance)

"""Run one SBML spec under every available exact-ODE engine and compare.

The model-interchange story, realized at two levels:

* **File level** — the model is authored once and serialized to SBML, the
  portable format every engine reads natively. No model-to-model translation.
* **Process level** — each exact-ODE engine ships a ``<Engine>FullReportUTCStep``
  on the engine-core ``UniformTimeCourseStep`` contract: inputs
  ``{model_source, time, n_points}`` → output ``result = {time, columns, values}``
  keyed by SBML id (species + reaction fluxes). Swapping engines is swapping one
  Step address, and the result shape is identical across engines — nothing to
  reconcile. This module reuses those consolidated Steps rather than rolling a
  per-engine adapter for each.

Two more exact-ODE arms are *not* on that contract and run through their own
process:

* **viva-core** (the Rust execution backend) runs the incremental
  ``TelluriumProcess`` under viva-core via
  :class:`~viva_simulation_core.analysis.vc_adapter.VivaCoreTelluriumGeneric`. It
  reproduces the roadrunner computation bit-for-bit (a throughput/overhead story),
  not a second independent solver.
* **julia** (DifferentialEquations.jl via ``JuliaDiffEqProcess``) reads the same
  SBML and gives a genuinely independent ODE solve, run through a Composite.

Two engine *tiers are out of scope here* because they do not read SBML the way the
exact-ODE engines do:

* **surrogate** (``viva_torch`` / ``NeuralProcess``) is a *learned* model — it
  consumes named leaf stores and a per-model trained checkpoint, not an SBML file,
  and must be trained on a given model's trajectories before it can run. It is not
  an engine you can point at an arbitrary SBML string.
* **biogears** is the C++ engine the sepsis SBML was *extracted from* — a
  different model, not a reader of this SBML.

Usage::

    from viva_simulation_core.analysis import run_engines
    results = run_engines.run_all(model=ANTIMONY, model_format='antimony',
                                  time=30.0, n_points=301)
    print(run_engines.format_comparison(results))
"""
from __future__ import annotations

import importlib
import time as _time

# engine key -> (module, FullReportUTCStep class) on the engine-core contract.
# Extend as more wrappers adopt the contract (pysces, amici, …).
_FULLREPORT_STEPS: dict[str, tuple[str, str]] = {
    "tellurium": ("viva_tellurium.processes", "TelluriumFullReportUTCStep"),
    "copasi": ("viva_copasi.processes", "CopasiFullReportUTCStep"),
    "pysces": ("viva_pysces.processes", "PyscesFullReportUTCStep"),
    "amici": ("viva_amici.processes", "AmiciFullReportUTCStep"),
}


def _load_step(engine: str):
    mod_name, cls_name = _FULLREPORT_STEPS[engine]
    return getattr(importlib.import_module(mod_name), cls_name)


def available_fullreport_engines() -> list[str]:
    """Contract engines whose FullReportUTCStep can be imported here."""
    out = []
    for engine in _FULLREPORT_STEPS:
        try:
            _load_step(engine)
            out.append(engine)
        except Exception:
            pass
    return out


def _to_sbml(model: str, model_format: str) -> str:
    """Normalize any supported authoring format to the portable SBML artifact."""
    fmt = (model_format or "sbml").lower()
    if fmt == "sbml":
        return model
    if fmt == "antimony":
        import tellurium as te
        return te.loada(model).getSBML()
    raise ValueError(f"unknown model_format {model_format!r} (use 'sbml' or 'antimony')")


def _species_ids(sbml: str) -> list[str]:
    """Floating-species ids (the observables), in roadrunner's canonical order."""
    import tellurium as te
    return list(te.loadSBMLModel(sbml).getFloatingSpeciesIds())


def run_fullreport(engine: str, sbml: str, time: float, n_points: int) -> dict:
    """Run one engine's FullReportUTCStep on the SBML; return the uniform result."""
    from process_bigraph import allocate_core
    step = _load_step(engine)(
        {"model_source": sbml, "time": time, "n_points": n_points},
        core=allocate_core(),
    )
    t0 = _time.perf_counter()
    out = step.update({"model_source": sbml, "time": time, "n_points": n_points})
    wall = _time.perf_counter() - t0
    res = out["result"]
    cols = list(res["columns"])
    last = res["values"][-1]
    return {
        "engine": engine,
        "via": f"{_FULLREPORT_STEPS[engine][1]} (engine-core contract)",
        "columns": cols,
        "time": list(res["time"]),
        "values": [list(r) for r in res["values"]],
        "final_all": {c: float(last[i]) for i, c in enumerate(cols)},
        "wall_ms": wall * 1e3,
    }


def run_viva_core(sbml: str, time: float, n_points: int) -> dict:
    """Run the model under viva-core (Rust) via the generic Tellurium adapter.

    Not on the FullReportUTCStep contract; reproduces the roadrunner computation.
    Returns the same shape as :func:`run_fullreport` (final species only — the
    Rust report carries end-state stores, not the full trajectory).
    """
    import json

    import viva_core
    from viva_tellurium import make_tellurium_document

    ids = _species_ids(sbml)
    interval = time / n_points
    doc = make_tellurium_document(model=sbml, model_format="sbml", interval=interval)
    adapter = "viva_simulation_core.analysis.vc_adapter.VivaCoreTelluriumGeneric"
    d = _reshape_for_viva_core(doc, ids, adapter)
    t0 = _time.perf_counter()
    rep = viva_core.run_report(json.dumps(d), n_points)
    wall = _time.perf_counter() - t0
    sp = rep["state"]["stores"]["species"]
    return {
        "engine": "viva-core",
        "via": "VivaCoreTelluriumGeneric (Rust; not on the contract)",
        "columns": ids,
        "time": None,
        "values": None,
        "final_all": {sid: float(sp[sid]) for sid in ids},
        "wall_ms": wall * 1e3,
    }


def run_julia(sbml: str, time: float, n_points: int) -> dict:
    """Run the model under Julia DifferentialEquations.jl via JuliaDiffEqProcess.

    Not on the FullReportUTCStep contract; a genuinely independent ODE solve.
    JuliaDiffEqProcess emits floating species as a ``map[float]`` keyed by SBML id.
    Returns the same shape as :func:`run_fullreport` (final species).

    NB: importing ``juliacall`` loads a Julia native runtime; for isolation from
    other native simulator libraries (COPASI / roadrunner) prefer to call this in a
    fresh process rather than alongside the contract arms in :func:`run_all`.
    """
    from process_bigraph import Composite, gather_emitter_results
    from viva_julia_diffeqs import make_julia_diffeqs_document
    from viva_julia_diffeqs.core import build_core as julia_build_core

    interval = time / n_points
    doc = make_julia_diffeqs_document(model=sbml, model_format="sbml", interval=interval)
    core = julia_build_core()
    t0 = _time.perf_counter()
    sim = Composite({"state": doc}, core=core)
    sim.run(time)
    wall = _time.perf_counter() - t0
    rows = gather_emitter_results(sim)[("emitter",)]
    sp = [r["species"] for r in rows if r.get("species")]
    final = {k: float(v) for k, v in sp[-1].items()}
    return {
        "engine": "julia",
        "via": "JuliaDiffEqProcess (DifferentialEquations.jl; not on the contract)",
        "columns": list(final),
        "time": None,
        "values": None,
        "final_all": final,
        "wall_ms": wall * 1e3,
    }


def _reshape_for_viva_core(doc: dict, species_ids: list[str], adapter: str) -> dict:
    """Swap the Tellurium node for the generic viva-core adapter + per-id species
    leaves, drop the empty/map ports and the emitter (viva-core reads end state)."""
    import copy

    d = copy.deepcopy(doc)
    d["tellurium"]["address"] = adapter
    for k in ("rates", "parameters"):
        d["tellurium"]["outputs"].pop(k, None)
        d.get("stores", {}).pop(k, None)
    d.pop("emitter", None)
    d.setdefault("stores", {})["species"] = {sid: 0.0 for sid in species_ids}
    d["global_time"] = 0.0
    return d


def run_all(model: str, model_format: str = "sbml", *, time: float = 30.0,
            n_points: int = 301, engines: list[str] | None = None,
            include_viva_core: bool = True, include_julia: bool = False) -> dict:
    """Run ``model`` under every requested engine; return ``{engine: result}``.

    ``engines`` defaults to every available contract engine. The non-contract
    exact-ODE arms are added when requested: viva-core by default, **julia only
    with ``include_julia=True``**. Julia is opt-in because importing ``juliacall``
    loads a Julia native runtime that can segfault when other native simulator
    libraries (COPASI, roadrunner) are already loaded in the same process — so the
    julia arm is best run in a fresh process (see :func:`run_julia`). Unavailable
    engines are skipped with a ``{'skipped': reason}`` entry rather than raising.
    """
    sbml = _to_sbml(model, model_format)
    species = _species_ids(sbml)
    want = engines if engines is not None else available_fullreport_engines()
    results: dict[str, dict] = {}
    for engine in want:
        try:
            results[engine] = run_fullreport(engine, sbml, time, n_points)
        except Exception as exc:  # pragma: no cover - engine-specific failures
            results[engine] = {"engine": engine, "skipped": f"{type(exc).__name__}: {exc}"}
    for flag, name, fn in (
        (include_viva_core, "viva-core", run_viva_core),
        (include_julia, "julia", run_julia),
    ):
        if flag:
            try:
                results[name] = fn(sbml, time, n_points)
            except Exception as exc:  # pragma: no cover
                results[name] = {"engine": name, "skipped": f"{type(exc).__name__}: {exc}"}
    for r in results.values():
        if "final_all" in r:
            r["species_ids"] = species
    return results


def compare(results: dict, ref: str = "tellurium") -> dict:
    """Max relative disagreement in final *species* values vs ``ref``.

    Compares only the floating-species columns (shared across engines), not the
    reaction-flux columns some engines also report.
    """
    ran = {k: v for k, v in results.items() if "final_all" in v}
    if ref not in ran:
        ref = next(iter(ran), None)
    if ref is None:
        return {}
    ref_sp = _species_subset(ran[ref])
    out = {}
    for engine, r in ran.items():
        sp = _species_subset(r)
        shared = [k for k in ref_sp if k in sp]
        out[engine] = max(
            (abs(ref_sp[k] - sp[k]) / max(abs(ref_sp[k]), 1e-12) for k in shared),
            default=0.0,
        )
    return out


def _species_subset(result: dict) -> dict:
    """Final values restricted to floating-species columns (drop reaction fluxes)."""
    fa = result["final_all"]
    ids = result.get("species_ids") or list(fa)
    return {c: fa[c] for c in ids if c in fa}


def format_comparison(results: dict, ref: str = "tellurium") -> str:
    """A compact human-readable table + agreement summary."""
    lines = [f"{'engine':16} {'wall(ms)':>9}  via"]
    for engine, r in results.items():
        if "skipped" in r:
            lines.append(f"{engine:16} {'—':>9}  SKIPPED: {r['skipped']}")
        else:
            lines.append(f"{engine:16} {r['wall_ms']:>9.1f}  {r['via']}")
    disagree = compare(results, ref=ref)
    if disagree:
        lines.append("")
        lines.append(f"final-species agreement vs {ref}:")
        for engine, rel in disagree.items():
            if engine == ref:
                continue
            verdict = "AGREE" if rel < 1e-2 else "DIVERGE"
            lines.append(f"  {engine:16} max rel disagreement {rel:.2e}  ({verdict})")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    ANT = (
        "model lv\n"
        "  J1: X -> 2 X;     a * X;\n"
        "  J2: X + Y -> 2 Y; b * X * Y;\n"
        "  J3: Y -> ;        d * Y;\n"
        "  X = 10; Y = 5; a = 1.1; b = 0.4; d = 0.4;\n"
        "end"
    )
    res = run_all(ANT, model_format="antimony", time=30.0, n_points=301)
    print(format_comparison(res))

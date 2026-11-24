#!/usr/bin/env python3
"""Study 04: Tellurium vs COPASI on the SAME sepsis SBML — a second exact-SBML engine.

Tellurium (roadrunner/CVODE) is the workspace's main SBML engine. This runs the
identical BioGears-sepsis SBML (severe arm) under both roadrunner and a second,
independent exact-SBML integrator — COPASI (LSODA), driven through libCOPASI's C++
bindings directly (no basico/pandas layer). It records (a) cross-engine agreement on
the 9 shared observables and (b) the wall-clock cost.

Point: two independent exact-SBML integrators reproduce the same trajectory, so the
SBML result is engine-independent, not a roadrunner artifact — and the cost of that
agreement differs because the model is event-dense (the per-beat event).

Writes charts/copasi_vs_tellurium.json.
Run from the investigation dir:
    PYTHONPATH=<worktree> python studies/04-performance/charts/run_copasi_vs_tellurium.py --horizon-hours 2
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
INV = HERE.parents[2]
SBML = INV / "sources" / "biogears_sepsis.xml"
# shared observable name -> SBML id (matches viva_simulation_core.composites.sepsis_sbml)
OBS = {"SBP": "SBP", "DBP": "DBP", "MAP": "MAP", "HR": "HR", "CO": "COmeas",
       "SVR": "SVR", "blood_volume": "Vblood", "tissue_integrity": "TI", "blood_pathogen": "PB"}


# severity -> tissue-pathogen seed (sepsis_sbml: 1e6 / 5e6 / 2.5e7 for severity 1/2/3)
SEED = {1: 1e6, 2: 5e6, 3: 2.5e7}


def make_severe_sbml(severity, infstart):
    """Bake the infection into a temp SBML as the INITIAL tissue-pathogen value, and
    disable the E_infect seeding event (infectionStart -> huge). This makes both engines
    start identically infected and run the real severe dynamics, independent of
    per-engine handling of the E_infect event's pathogen-seeding assignment (which
    COPASI and roadrunner evaluate differently)."""
    import re, tempfile
    s = SBML.read_text()
    seed = SEED.get(int(severity), 2.5e7)
    # Replicate the post-E_infect state as initial conditions: seed PT, switch the
    # infection on (infOn=1, TLR=1), and disable the E_infect event so it does not
    # re-seed. Both engines then start identically infected.
    def setval(src, pid, val):
        return re.sub(r'(id="%s"[^>]*\bvalue=")[^"]*(")' % pid,
                      lambda m: m.group(1) + str(val) + m.group(2), src, count=1)
    s = setval(s, "PT", seed)
    s = setval(s, "infOn", 1)
    s = setval(s, "TLR", 1)
    s = setval(s, "infectionStart", "1e12")
    f = tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False)
    f.write(s); f.close()
    return f.name


def run_tellurium(sbml_path, tend, n):
    import roadrunner
    rr = roadrunner.RoadRunner(sbml_path)
    rr.getIntegrator().setValue("relative_tolerance", 1e-6)
    rr.getIntegrator().setValue("absolute_tolerance", 1e-10)
    rr.reset()
    sel = ["time"] + [OBS[k] for k in OBS]
    t0 = time.perf_counter()
    d = np.array(rr.simulate(0, tend, n, sel))
    return time.perf_counter() - t0, d


def run_copasi(sbml_path, tend, n):
    import COPASI
    dm = COPASI.CRootContainer.addDatamodel(); dm.importSBML(sbml_path); model = dm.getModel()
    task = dm.getTask("Time-Course"); task.setMethodType(COPASI.CTaskEnum.Method_deterministic)
    prob = task.getProblem(); prob.setDuration(float(tend)); prob.setStepNumber(n - 1)
    prob.setTimeSeriesRequested(True); prob.setOutputStartTime(0.0); prob.setAutomaticStepSize(False)
    task.setScheduled(True); m = task.getMethod()
    for pn, v in (("Relative Tolerance", 1e-6), ("Absolute Tolerance", 1e-10), ("Max Internal Steps", 1000000)):
        p = m.getParameter(pn);  p.setValue(float(v)) if p else None
    task.initializeRaw(COPASI.CCopasiTask.OUTPUT_UI)
    t0 = time.perf_counter(); task.processRaw(True); wall = time.perf_counter() - t0
    ts = task.getTimeSeries(); col = {ts.getTitle(i): i for i in range(ts.getNumVariables())}
    steps = ts.getRecordedSteps()
    out = {"time": [ts.getData(s, col.get("Time", 0)) for s in range(steps)]}
    for k, sid in OBS.items():
        c = col.get(sid) or col.get("Values[%s]" % sid)
        out[k] = [ts.getData(s, c) for s in range(steps)] if c is not None else None
    return wall, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon-hours", type=float, default=2.0)
    ap.add_argument("--interval-s", type=float, default=60.0)
    ap.add_argument("--severity", type=int, default=3)
    args = ap.parse_args()
    tend = args.horizon_hours * 3600.0
    n = int(round(tend / args.interval_s)) + 1

    severe_sbml = make_severe_sbml(args.severity, 1.0)
    print(f"[cmp] Tellurium (roadrunner) severe {args.horizon_hours}h...", flush=True)
    te_wall, te = run_tellurium(severe_sbml, tend, n)
    print(f"[cmp]   wall={te_wall:.1f}s", flush=True)
    print(f"[cmp] COPASI (direct libCOPASI) severe {args.horizon_hours}h...", flush=True)
    cp_wall, cp = run_copasi(severe_sbml, tend, n)
    print(f"[cmp]   wall={cp_wall:.1f}s", flush=True)

    # align by index (same uniform time grid), compute cross-engine agreement
    agree = {}
    for i, k in enumerate(OBS, start=1):
        a = np.asarray(te[:, i], float)
        b = np.asarray(cp[k], float)[: len(a)]
        scale = max(np.abs(a).max(), 1e-9)
        rel = float(np.max(np.abs(a[: len(b)] - b)) / scale)
        agree[k] = {"max_rel_diff": rel, "tellurium_end": float(a[-1]), "copasi_end": float(b[-1])}
    out = {"meta": {"horizon_h": args.horizon_hours, "interval_s": args.interval_s, "severity": args.severity,
                    "engines": {"tellurium": "roadrunner/CVODE (rtol 1e-6)",
                                "copasi": "libCOPASI LSODA direct C++ (rtol 1e-6)"}},
           "wall": {"tellurium_s": te_wall, "copasi_s": cp_wall, "copasi_over_tellurium": cp_wall / te_wall},
           "agreement": agree,
           "trajectories": {"time_h": (te[:, 0] / 3600.0).tolist(),
                            **{f"tellurium_{k}": te[:, i].tolist() for i, k in enumerate(OBS, start=1)},
                            **{f"copasi_{k}": cp[k] for k in OBS}}}
    (HERE / "copasi_vs_tellurium.json").write_text(json.dumps(out, indent=1))
    worst = max(agree.items(), key=lambda kv: kv[1]["max_rel_diff"])
    print(f"[cmp] wall: Tellurium {te_wall:.1f}s vs COPASI {cp_wall:.1f}s ({cp_wall/te_wall:.1f}x)", flush=True)
    print(f"[cmp] max cross-engine rel diff across 9 observables: {worst[1]['max_rel_diff']:.2e} ({worst[0]})", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Integrator benchmark for the BioGears sepsis SBML extraction.

Question: would reimplementing the ODE *integrator* (e.g. in Rust, as viva-hybrid
does with a fixed-step RK4) speed up this model versus the libroadrunner (CVODE)
path we already use?

Method: run the real 91-state model under roadrunner's integrators on a short
representative window:
  - cvode : stiff BDF (compiled C/SUNDIALS) -- the method libroadrunner uses by default
  - rk45  : adaptive explicit (Dormand-Prince)
  - rk4   : fixed-step explicit -- the SAME algorithm as viva-hybrid's engine
We sweep the RK4 step size, measure stability + accuracy vs the stiff reference,
and extrapolate the step count to a full 48 h simulation (172800 s).

Writes studies/06-ode-integrator-runtime/out/integrator_bench.json.
Run: .venv/bin/python workspace/investigations/biogears-sbml-sepsis/bench_integrators.py
"""
import json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import roadrunner

HERE = Path(__file__).resolve().parent
MODEL = HERE / "sources" / "biogears_sepsis.xml"
OUT = HERE / "studies" / "06-ode-integrator-runtime" / "out" / "integrator_bench.json"
T = 60.0                 # short window; stiffness (fast eigenvalue) sets the stability limit regardless of window
H48 = 172800.0           # 48 h in seconds, for step-count extrapolation
RK45_CAP_S = 45          # wall-clock cap for the stability-limited explicit run

def load():
    rr = roadrunner.RoadRunner(str(MODEL))
    return rr

def cvode_ref(rr):
    rr.setIntegrator("cvode"); ig = rr.getIntegrator()
    ig.setValue("stiff", True); ig.setValue("relative_tolerance", 1e-9); ig.setValue("absolute_tolerance", 1e-13)
    rr.reset(); res = rr.simulate(0, T, 601)
    return np.array(res[-1, 1:], dtype=float)

def timed(rr, integrator, npoints):
    rr.setIntegrator(integrator)
    if integrator == "cvode":
        rr.getIntegrator().setValue("stiff", True)
    rr.reset()
    t0 = time.perf_counter()
    res = rr.simulate(0, T, npoints)
    wall = time.perf_counter() - t0
    final = np.array(res[-1, 1:], dtype=float)
    return bool(np.all(np.isfinite(final))), wall, final

def rk45_subprocess():
    """rk45 is stability-limited on this stiff model; run it with a hard cap so it can't hang."""
    code = (
        "import roadrunner,numpy as np,time;"
        f"rr=roadrunner.RoadRunner(r'{MODEL}');rr.setIntegrator('rk45');rr.reset();"
        "t0=time.perf_counter();res=rr.simulate(0,%g,61);print('%%.4f'%%(time.perf_counter()-t0))" % T
    )
    try:
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=RK45_CAP_S)
        return {"completed": True, "wall_s": float(out.stdout.strip().splitlines()[-1])}
    except subprocess.TimeoutExpired:
        return {"completed": False, "cap_s": RK45_CAP_S}

def main():
    rr = load()
    ids = list(rr.model.getRateRuleSymbols())
    n_states = len(ids)

    ref = cvode_ref(rr)
    scale = np.maximum(np.abs(ref), 1e-12 + 1e-9 * np.nanmax(np.abs(ref)))
    def maxrel(f):
        return float(np.max(np.abs(f - ref) / scale))

    # cvode baseline (default tol, matched output grid)
    ok, wc, fc = timed(rr, "cvode", 61)
    cvode = {"wall_s": wc, "ok": ok, "max_relerr": maxrel(fc)}

    # rk4 fixed-step sweep (= viva-hybrid's integrator)
    rk4 = []
    for dt in (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001):
        npoints = int(round(T / dt)) + 1
        ok, w, f = timed(rr, "rk4", npoints)
        rk4.append({"dt_s": dt, "steps_48h": H48 / dt, "wall_s": w, "finite": ok,
                    "max_relerr": (maxrel(f) if ok else float("inf"))})

    # per-state diagnostic at dt=0.001 (173 M steps / 48 h): which states blow up?
    ok, _, f001 = timed(rr, "rk4", int(round(T / 0.001)) + 1)
    den = np.maximum(np.abs(ref), 1e-30)
    rel = np.abs(f001 - ref) / den
    nn = np.abs(ref) > 1e-6 * np.max(np.abs(ref))   # "non-negligible" states
    worst = np.argsort(rel)[::-1][:8]
    per_state = {
        "dt_s": 0.001, "steps_48h": H48 / 0.001,
        "n_states": n_states,
        "n_over_1pct": int(np.sum(rel > 0.01)),
        "n_over_100pct": int(np.sum(rel > 1.0)),
        "median_relerr_nonneg": float(np.median(rel[nn])),
        "p90_relerr_nonneg": float(np.percentile(rel[nn], 90)),
        "worst": [{"id": ids[i], "cvode": float(ref[i]), "rk4": float(f001[i]), "relerr": float(rel[i])} for i in worst],
        # full per-state error so the figure is a pure read (no re-simulation needed)
        "all_states": [{"id": ids[i], "relerr": float(rel[i])} for i in range(n_states)],
    }

    result = {
        "model": "biogears_sepsis.xml", "n_states": n_states, "window_s": T, "horizon_48h_s": H48,
        "cvode_ref_tol": {"rtol": 1e-9, "atol": 1e-13},
        "cvode": cvode,            # stiff BDF, compiled C -- the default libroadrunner path
        "rk45": rk45_subprocess(), # adaptive explicit
        "rk4": rk4,                # fixed-step explicit == viva-hybrid
        "per_state_dt001": per_state,
        "note": "cvode is libroadrunner's compiled-C SUNDIALS BDF. rk4 is the same algorithm as the viva-hybrid Rust engine.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print("wrote", OUT)
    print("cvode %.3fs | rk45 %s | rk4 finite only for dt<=%.3g | dt=0.001: %d/%d states >1%%, %d >100%%" % (
        cvode["wall_s"],
        ("%.3fs" % result["rk45"]["wall_s"]) if result["rk45"]["completed"] else (">%ds (DNF)" % RK45_CAP_S),
        max([r["dt_s"] for r in rk4 if r["finite"]]),
        per_state["n_over_1pct"], n_states, per_state["n_over_100pct"]))

if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Study 08: event-free (reduced) vs event-driven (full) sepsis model.

Runs the event-free model (sources/eventfree_sepsis.ant) under roadrunner and
COPASI, compares its trajectory to the committed full-model exact severe/healthy
arms (data/real_trajectories.json), and records the speed gap. Writes
out/comparison.json.

Run from the investigation dir:
    PYTHONPATH=<worktree> python studies/08-reduced-model/run_comparison.py
"""
from __future__ import annotations

import json, time, os
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
INV = HERE.parents[1]
ANT = (HERE / "sources" / "eventfree_sepsis.ant").read_text()
REF = json.loads((INV / "data" / "real_trajectories.json").read_text())
OBS = {"MAP": "MAP", "SVR": "SVR", "blood_pathogen": "PB", "tissue_integrity": "TI"}

# Full-model runtime: roadrunner from study 04 (committed 48 h wall); COPASI from
# our direct-libCOPASI measurement (~4.6 s / 60 s -> ~3.7 h / 48 h).
FULL_WALL = {"roadrunner_h": 2.81, "copasi_h": 3.7}


def run_eventfree_rr(seed, tend_h=48.0, n=193):
    import tellurium as te
    r = te.loadAntimonyModel(ANT); r.setValue("seed", float(seed)); r.reset()
    t0 = time.perf_counter()
    d = r.simulate(0, tend_h * 3600, n, ["time", "MAP", "SVR", "PB", "TI"])
    return time.perf_counter() - t0, np.array(d)


def time_eventfree_copasi(seed, tend_h=48.0):
    import COPASI, tellurium as te
    r = te.loadAntimonyModel(ANT); r.setValue("seed", float(seed))
    sbml = r.getSBML()
    dm = COPASI.CRootContainer.addDatamodel(); dm.importSBMLFromString(sbml)
    task = dm.getTask("Time-Course"); task.setMethodType(COPASI.CTaskEnum.Method_deterministic)
    prob = task.getProblem(); prob.setDuration(tend_h * 3600.0); prob.setStepNumber(192)
    prob.setTimeSeriesRequested(True); prob.setOutputStartTime(0.0); prob.setAutomaticStepSize(False)
    task.setScheduled(True)
    task.initializeRaw(COPASI.CCopasiTask.OUTPUT_UI)
    t0 = time.perf_counter(); task.processRaw(True); return time.perf_counter() - t0


def main():
    (HERE / "out").mkdir(exist_ok=True)
    out = {"full_model_wall": FULL_WALL, "arms": {}, "speed": {}}

    # severe seed is baked into the .ant; healthy = seed 0
    seeds = {"severe": None, "healthy": 0.0}
    r_import = __import__("tellurium").loadAntimonyModel(ANT)
    seeds["severe"] = float(r_import.getValue("seed"))

    for arm, seed in seeds.items():
        w, A = run_eventfree_rr(seed)
        ref = REF[f"exact_sev{'3' if arm=='severe' else '0'}"]
        tref_h = np.array(ref["t_h"])
        t_ef = A[:, 0] / 3600.0
        rec = {"t_h": t_ef.tolist(), "roadrunner_wall_s": w}
        fid = {}
        for i, (rk, mk) in enumerate(OBS.items(), start=1):
            ef = A[:, i]
            rf = np.interp(t_ef, tref_h, np.array(ref["obs"][rk]))
            rec[mk] = ef.tolist(); rec[f"ref_{mk}"] = rf.tolist()
            rmse = float(np.sqrt(np.mean((ef - rf) ** 2)))
            fid[mk] = {"rmse": rmse, "pct_of_max": 100 * rmse / max(abs(rf).max(), 1e-9),
                       "ef_end": float(ef[-1]), "ref_end": float(rf[-1])}
        rec["fidelity"] = fid
        out["arms"][arm] = rec

    # speed: event-free under both engines (severe), vs cited full-model wall
    sev = seeds["severe"]
    ef_rr = min(run_eventfree_rr(sev)[0] for _ in range(3))
    ef_cp = min(time_eventfree_copasi(sev) for _ in range(3))
    out["speed"] = {"eventfree_roadrunner_s": ef_rr, "eventfree_copasi_s": ef_cp,
                    "full_roadrunner_s": FULL_WALL["roadrunner_h"] * 3600,
                    "full_copasi_s": FULL_WALL["copasi_h"] * 3600,
                    "speedup_roadrunner": FULL_WALL["roadrunner_h"] * 3600 / ef_rr,
                    "speedup_copasi": FULL_WALL["copasi_h"] * 3600 / ef_cp}
    (HERE / "out" / "comparison.json").write_text(json.dumps(out, indent=1))
    print("event-free severe:", {k: round(v["ef_end"], 2) for k, v in out["arms"]["severe"]["fidelity"].items()})
    print("fidelity (%% of max):", {k: round(v["pct_of_max"], 1) for k, v in out["arms"]["severe"]["fidelity"].items()})
    print("speed: event-free rr=%.1g s cp=%.1g s | full rr=%.0f s cp=%.0f s | speedup ~%.0gx / ~%.0gx"
          % (ef_rr, ef_cp, out["speed"]["full_roadrunner_s"], out["speed"]["full_copasi_s"],
             out["speed"]["speedup_roadrunner"], out["speed"]["speedup_copasi"]))


if __name__ == "__main__":
    main()

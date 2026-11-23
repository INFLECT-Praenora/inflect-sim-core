#!/usr/bin/env python
"""Study 06 upgrade: treatment counterfactuals + a coupling-step convergence test.

Two experiments on the sepsis_cosim composite (severity 3):

  COUNTERFACTUALS (out/cosim_counterfactuals.json): three arms from the identical
  composite, differing only in treatment timing —
    * untreated       (treat=False)
    * early treatment (treat=True, start_time_s=0)
    * delayed treatment (treat=True, start_time_s=DELAY)
  so we can compare pathogen / tissue integrity / MAP under each.

  CONVERGENCE (out/cosim_convergence.json): the early-treatment arm run at several
  coupling step sizes (interval_s). As the step shrinks, the one-tick coupling lag
  shrinks, so the endpoint should converge — a numerical-consistency check on the
  co-simulation, separate from any pharmacological claim.

Run from the investigation dir:
    PYTHONPATH=<worktree> python studies/06-cosimulation/run_counterfactuals.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OBS = ["blood_pathogen", "tissue_integrity", "MAP", "SVR"]


def run_arm(treat, start_time_s, horizon_h, interval_s, severity=3):
    from process_bigraph import Composite, gather_emitter_results
    from process_bigraph.emitter import RAMEmitter
    from viva_simulation_core.core import build_core
    from viva_simulation_core.composites.sepsis_cosim import (
        make_sepsis_cosim_composite, extract_cosim)

    core = build_core()
    core.register_link("ram-emitter", RAMEmitter)
    doc = make_sepsis_cosim_composite(severity=severity, treat=treat, interval_s=interval_s,
                                      start_time_s=start_time_s, horizon_s=horizon_h * 3600.0)
    sim = Composite({"state": doc}, core=core)
    t0 = time.perf_counter()
    for _ in range(max(1, int(round(horizon_h * 3600.0 / interval_s)))):
        sim.run(interval_s)
    wall = time.perf_counter() - t0
    d = extract_cosim(gather_emitter_results(sim)[("emitter",)])
    arm = {"time_h": [t / 3600.0 for t in d["time"]], "drug": d["drug"],
           "treating": d["treating"], "wall_s": wall}
    for k in OBS:
        arm[k] = d[k]
    return arm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon-hours", type=float, default=2.0)
    ap.add_argument("--interval-s", type=float, default=120.0)
    ap.add_argument("--delay-hours", type=float, default=1.0)
    args = ap.parse_args()
    out = HERE / "out"
    out.mkdir(parents=True, exist_ok=True)

    # --- Counterfactuals ---
    print("[cf] untreated...", flush=True)
    untreated = run_arm(False, 0.0, args.horizon_hours, args.interval_s)
    print("[cf] early treatment...", flush=True)
    early = run_arm(True, 0.0, args.horizon_hours, args.interval_s)
    print(f"[cf] delayed treatment (start {args.delay_hours} h)...", flush=True)
    delayed = run_arm(True, args.delay_hours * 3600.0, args.horizon_hours, args.interval_s)
    cf = {"meta": {"severity": 3, "horizon_h": args.horizon_hours,
                   "interval_s": args.interval_s, "delay_h": args.delay_hours},
          "untreated": untreated, "early": early, "delayed": delayed}
    (out / "cosim_counterfactuals.json").write_text(json.dumps(cf, indent=1))
    for tag, a in (("untreated", untreated), ("early", early), ("delayed", delayed)):
        print(f"[cf]   {tag:9s} PB_end={a['blood_pathogen'][-1]:.3g} "
              f"TI_end={a['tissue_integrity'][-1]:.4f} wall={a['wall_s']:.0f}s", flush=True)

    # --- Coupling-step convergence (early treatment, 1 h) ---
    conv = {"meta": {"severity": 3, "horizon_h": 1.0, "arm": "early"}, "runs": []}
    for iv in (240.0, 120.0, 60.0):
        print(f"[conv] interval={iv:.0f}s...", flush=True)
        a = run_arm(True, 0.0, 1.0, iv)
        conv["runs"].append({"interval_s": iv,
                             "blood_pathogen_end": a["blood_pathogen"][-1],
                             "tissue_integrity_end": a["tissue_integrity"][-1],
                             "MAP_end": a["MAP"][-1], "wall_s": a["wall_s"]})
        print(f"[conv]   interval={iv:.0f}s PB_end={a['blood_pathogen'][-1]:.4g} "
              f"TI_end={a['tissue_integrity'][-1]:.5f}", flush=True)
    (out / "cosim_convergence.json").write_text(json.dumps(conv, indent=1))
    pbs = [r["blood_pathogen_end"] for r in conv["runs"]]
    print(f"[conv] PB_end across steps {[round(x,4) for x in pbs]} "
          f"(spread {max(pbs)-min(pbs):.4g})", flush=True)
    print("[done] wrote out/cosim_counterfactuals.json and out/cosim_convergence.json", flush=True)


if __name__ == "__main__":
    main()

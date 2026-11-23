#!/usr/bin/env python
"""CO-SIMULATION arm runner (study 06): the sepsis physiology engine coupled to a
closed-loop antibiotic controller, run as ONE process-bigraph composite in which
the two processes exchange state every tick.

Runs two matched arms of the `sepsis_cosim` composite at severity 3:
  * untreated  (treat=False -> controller inert = matched control)
  * treated    (treat=True  -> closed-loop antibiotic)
and writes out/cosim_arms.json = {meta, untreated:{...}, treated:{...}} where each
arm is {time_h, blood_pathogen, tissue_integrity, MAP, SVR, HR, drug, treating}.

Run from the investigation dir (engines live in the canonical venv):
    PYTHONPATH=<worktree> python studies/06-cosimulation/run_cosim.py --horizon-hours 4

The exact engine is slow (~minutes per simulated hour), so the default horizon is
short: it captures the controlled variable (pathogen clearance) and the live drug
PK. The downstream clinical collapse is a 37-72 h process (studies 03/05); this run
shows the loop removing its driver, not the multi-day endpoint.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OBS = ["blood_pathogen", "tissue_integrity", "MAP", "SVR", "HR"]


def run_arm(treat: bool, severity: int, horizon_h: float, interval_s: float) -> dict:
    from process_bigraph import Composite, gather_emitter_results
    from process_bigraph.emitter import RAMEmitter
    from viva_simulation_core.core import build_core
    from viva_simulation_core.composites.sepsis_cosim import (
        make_sepsis_cosim_composite, extract_cosim)

    core = build_core()
    core.register_link("ram-emitter", RAMEmitter)
    doc = make_sepsis_cosim_composite(severity=severity, treat=treat,
                                      interval_s=interval_s, horizon_s=horizon_h * 3600.0)
    sim = Composite({"state": doc}, core=core)
    n = max(1, int(round(horizon_h * 3600.0 / interval_s)))
    t0 = time.perf_counter()
    for _ in range(n):
        sim.run(interval_s)
    wall = time.perf_counter() - t0
    d = extract_cosim(gather_emitter_results(sim)[("emitter",)])
    arm = {"time_h": [t / 3600.0 for t in d["time"]],
           "drug": d["drug"], "treating": d["treating"], "wall_s": wall}
    for k in OBS:
        arm[k] = d[k]
    return arm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--severity", type=int, default=3)
    ap.add_argument("--horizon-hours", type=float, default=2.0)
    ap.add_argument("--interval-s", type=float, default=120.0)
    ap.add_argument("--out", type=Path, default=HERE / "out" / "cosim_arms.json")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(severity=args.severity, horizon_h=args.horizon_hours,
                interval_s=args.interval_s, composite="sepsis_cosim")
    print(f"[cosim] untreated arm (severity={args.severity}, {args.horizon_hours}h)...", flush=True)
    untreated = run_arm(False, args.severity, args.horizon_hours, args.interval_s)
    print(f"[cosim]   untreated wall={untreated['wall_s']:.0f}s "
          f"blood_pathogen {untreated['blood_pathogen'][0]:.3g}->{untreated['blood_pathogen'][-1]:.3g}", flush=True)
    print(f"[cosim] treated arm (closed-loop antibiotic)...", flush=True)
    treated = run_arm(True, args.severity, args.horizon_hours, args.interval_s)
    print(f"[cosim]   treated   wall={treated['wall_s']:.0f}s "
          f"blood_pathogen {treated['blood_pathogen'][0]:.3g}->{treated['blood_pathogen'][-1]:.3g} "
          f"drug_end={treated['drug'][-1]:.3g}", flush=True)

    args.out.write_text(json.dumps({"meta": meta, "untreated": untreated, "treated": treated}, indent=1))
    pb_u = untreated["blood_pathogen"][-1]
    pb_t = treated["blood_pathogen"][-1]
    ratio = (pb_u / pb_t) if pb_t > 1e-9 else float("inf")
    print(f"[cosim] wrote {args.out}  |  final bacteremia untreated/treated = {ratio:.1f}x", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""One Nextflow SEPSIS_RUN task: build and run the REAL sepsis_sbml composite at a
given severity and print its final clinical observables.

Used by sepsis.nf — this is the vivarium composite that runs inside each Nextflow
task. It uses the simulation core's own build_core + composite exactly as the rest
of the workspace does; the observables live in the composite's emitter, so we read
them back with gather_emitter_results (the CLI `--emit` only sees top-level state).

    python run_sepsis_task.py --severity 3 --steps 360
    -> MAP=72.31 TI=0.68 PB=120.4
"""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--severity", type=int, required=True)
    ap.add_argument("--steps", type=int, default=360)
    ap.add_argument("--interval-s", type=float, default=10.0)
    args = ap.parse_args()

    from process_bigraph import Composite, gather_emitter_results
    from process_bigraph.emitter import RAMEmitter
    from viva_simulation_core.core import build_core
    from viva_simulation_core.composites.sepsis_sbml import (
        make_sepsis_sbml_composite, SEPSIS_OBSERVABLES)

    core = build_core()
    core.register_link("ram-emitter", RAMEmitter)
    doc = make_sepsis_sbml_composite(severity=args.severity, infection_start_s=0,
                                     horizon_s=args.steps * args.interval_s,
                                     interval_s=args.interval_s)
    sim = Composite({"state": doc}, core=core)
    for _ in range(args.steps):
        sim.run(args.interval_s)

    rows = gather_emitter_results(sim)[("emitter",)]
    params = next(r["parameters"] for r in reversed(rows) if r.get("parameters"))
    mapv = params[SEPSIS_OBSERVABLES["MAP"]]
    ti = params[SEPSIS_OBSERVABLES["tissue_integrity"]]
    pb = params[SEPSIS_OBSERVABLES["blood_pathogen"]]
    print(f"MAP={mapv:.2f} TI={ti:.4f} PB={pb:.4g}")


if __name__ == "__main__":
    main()

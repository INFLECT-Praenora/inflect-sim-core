#!/usr/bin/env python
"""Demonstrate the simulation-core contract: model in, results + verification out.

Runs several models through the one interface — simulate(model, request) — and
records, for each, the results it returned and the verification it carries. One
interface, different model forms and engines, every run self-verified.

Writes out/contract_runs.json. Run from the investigation dir:
    PYTHONPATH=<worktree> python studies/00-contract/run_contract.py
"""
from __future__ import annotations

import json
from pathlib import Path

from viva_simulation_core import simulate, SimulationRequest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
SBML = str(ROOT / "workspace/investigations/biogears-sbml-sepsis/sources/biogears_sepsis.xml")

CASES = [
    ("SBML file → Tellurium",
     lambda: simulate(SBML, duration=600, interval=60, observables=["MAP", "SVR", "PB", "TI"])),
    ("same model + cross-engine verification (COPASI)",
     lambda: simulate(SBML, duration=600, interval=60, observables=["MAP", "SVR", "PB", "TI"],
                      cross_engine="copasi")),
    ("SBML file → COPASI engine",
     lambda: simulate(SBML, duration=600, interval=60, observables=["MAP", "SVR", "PB", "TI"],
                      engine="copasi")),
    ("registered composite id (severity 1)",
     lambda: simulate("viva_simulation_core.composites.sepsis_sbml.sepsis_sbml",
                      SimulationRequest(duration=300, interval=30, observables=["MAP", "SVR", "TI"],
                                        parameters={"severity": 1}))),
]


def main() -> None:
    (HERE / "out").mkdir(exist_ok=True)
    runs = []
    for label, fn in CASES:
        print(f"[contract] {label} ...", flush=True)
        out = fn()
        v = out.verification
        rec = {
            "label": label, "engine": out.engine, "wall_s": out.wall_s,
            "n_points": len(out.results.get("time", [])),
            "results_end": {k: (s[-1] if s else None) for k, s in out.results.items() if k != "time"},
            "verification": {"passed": v.passed, "ran": v.ran, "finite": v.finite,
                             "admissible": v.admissible, "cross_engine": v.cross_engine},
            "provenance": out.provenance,
            # keep one run's trajectory for the results panel
            "trajectory": ({"time_h": [t / 3600 for t in out.results["time"]],
                            **{k: out.results[k] for k in ("MAP", "SVR", "PB", "TI") if k in out.results}}
                           if label.startswith("same model") else None),
        }
        runs.append(rec)
        print(f"[contract]   verification passed={v.passed} "
              f"cross={'-' if not v.cross_engine else ('%.1e' % v.cross_engine['max_rel_diff'])}", flush=True)
    (HERE / "out" / "contract_runs.json").write_text(json.dumps({"runs": runs}, indent=1))
    npass = sum(1 for r in runs if r["verification"]["passed"])
    print(f"[contract] wrote out/contract_runs.json — {npass}/{len(runs)} runs verified", flush=True)


if __name__ == "__main__":
    main()

"""Same-harness comparison: event-free proto_valve vs the full sepsis_sbml model.

Runs BOTH models through the identical process-bigraph Composite harness (same
interval, same emitter, same machine) at severity 3, so the wall-clock *ratio* and
the per-observable agreement are apples-to-apples. This is the measured on-machine
basis for the proto_valve arm in study 04 (speed) and study 05 (fidelity).

Why not reuse the prototype report's numbers directly: those were single runs on a
different sandbox driving roadrunner's raw wheel (one long simulate() call), ~17-21 s
for 30 min. The Composite drives the model in interval_s chunks with a reset +
parameter re-apply each step, so its absolute wall-clock is higher — but both models
pay that overhead identically, so only the ratio measured *here* is comparable to the
other arms in this study.

Writes proto_valve_compare.json (time series for all 9 observables + timings).
Run from this dir:  python run_proto_valve_compare.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from process_bigraph import Composite, allocate_core, gather_emitter_results
from process_bigraph.emitter import RAMEmitter
from viva_tellurium.processes import TelluriumProcess

from viva_simulation_core.composites.sepsis_sbml import (
    make_sepsis_sbml_composite, extract_observables, SEPSIS_OBSERVABLES)
from viva_simulation_core.composites.sepsis_sbml_eventfree_valve import (
    make_sepsis_sbml_eventfree_valve_composite as make_eventfree)

HERE = Path(__file__).parent
SEVERITY = 3
INFECTION_START_S = 1.0
INTERVAL_S = 10.0
HORIZON_S = 7200.0   # 2 h model time — past the 30-min near-baseline window


def _run(doc, t):
    core = allocate_core()
    core.register_link('TelluriumProcess', TelluriumProcess)
    core.register_link('ram-emitter', RAMEmitter)
    sim = Composite({'state': doc}, core=core)
    t0 = time.perf_counter()
    sim.run(t)
    wall = time.perf_counter() - t0
    rows = gather_emitter_results(sim)[('emitter',)]
    obs = extract_observables(rows)
    # recover the time axis from the emitter rows (global_time)
    tvec = [float(r.get('global_time', i * INTERVAL_S)) for i, r in enumerate(rows)
            if r.get('parameters')]
    return obs, tvec, wall


def main():
    full_doc = make_sepsis_sbml_composite(
        severity=SEVERITY, infection_start_s=INFECTION_START_S, interval_s=INTERVAL_S)
    pv_doc = make_eventfree(
        severity=SEVERITY, infection_start_s=INFECTION_START_S, interval_s=INTERVAL_S)

    print(f"running full sepsis_sbml (severity {SEVERITY}, {HORIZON_S:.0f}s) ...")
    full_obs, tvec, full_wall = _run(full_doc, HORIZON_S)
    print(f"  full: {full_wall:.2f}s")
    print(f"running event-free proto_valve (severity {SEVERITY}, {HORIZON_S:.0f}s) ...")
    pv_obs, tvec_pv, pv_wall = _run(pv_doc, HORIZON_S)
    print(f"  proto_valve: {pv_wall:.2f}s  ({full_wall / pv_wall:.3f}x faster)")

    # per-observable agreement over the matched time axis
    n = min(len(tvec), len(tvec_pv))
    agreement = {}
    for k in SEPSIS_OBSERVABLES:
        a = full_obs[k][:n]
        b = pv_obs[k][:n]
        max_abs = max(abs(x - y) for x, y in zip(a, b))
        denom = max(abs(x) for x in a) or 1.0
        agreement[k] = {"max_abs_diff": max_abs, "max_rel_diff": max_abs / denom,
                        "final_full": a[-1], "final_pv": b[-1]}

    out = {
        "scenario": {"severity": SEVERITY, "infection_start_s": INFECTION_START_S,
                     "interval_s": INTERVAL_S, "horizon_s": HORIZON_S},
        "harness": "process-bigraph Composite (same interval/emitter/machine for both arms)",
        "timing": {"full_wall_s": full_wall, "proto_valve_wall_s": pv_wall,
                   "speedup": full_wall / pv_wall},
        "time_h": [t / 3600.0 for t in tvec[:n]],
        "full": {k: full_obs[k][:n] for k in SEPSIS_OBSERVABLES},
        "proto_valve": {k: pv_obs[k][:n] for k in SEPSIS_OBSERVABLES},
        "agreement": agreement,
    }
    (HERE / "proto_valve_compare.json").write_text(json.dumps(out, indent=1))
    print("wrote proto_valve_compare.json")
    print("agreement (max abs diff):")
    for k, v in agreement.items():
        print(f"  {k:18s} {v['max_abs_diff']:.4g}  (rel {v['max_rel_diff']:.2g})")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Rust execution-backend arm (study 04): run the SAME sepsis composite under the
process-bigraph Python engine and under the Rust viva-core backend, and record
whether they agree and what the engine swap costs.

This surfaces the Rust backend as a runnable artifact rather than a prose aside:
viva-core executes the composite graph (state, scheduling, emit) while roadrunner
does the same stiff ODE math, so the swap is an EXECUTION-BACKEND swap. On a single
heavy stiff process the wall difference measures framework overhead, not a speedup —
viva-core's per-tick win shows on many-process / many-tick composites, and the stiff
solve itself is immovable (see the stiff-ODE finding). The value demonstrated here is
backend interchangeability + determinism: same composite, two engines, same numbers.

Writes rust_backend.json next to this script. Run from the investigation dir:
    PYTHONPATH=<worktree> python studies/04-performance/charts/run_rust_backend.py
"""
from __future__ import annotations

import json
from pathlib import Path

from viva_simulation_core.analysis.run_s4_engines import run_under_both

HERE = Path(__file__).resolve().parent


def main() -> None:
    # Healthy baseline (severity 0): fast, deterministic, no infection event — a clean
    # determinism check of the two execution backends on the real 91-state model.
    r = run_under_both(severity=0, n_ticks=60, interval_s=10.0)
    out = {
        "verdict": r["verdict"],                       # bit-identical | within-tolerance | DIVERGENT
        "abs_diff": r["abs_diff"],
        "python_wall_s": r["python"]["wall_s"],
        "viva_core_wall_s": r["viva_core"]["wall_s"],
        "wall_ratio_python_over_rust": r["wall_ratio_python_over_rust"],
        "n_ticks": r["n_ticks"], "interval_s": r["interval_s"], "severity": r["severity"],
        "note": ("Same sepsis composite under process-bigraph (Python) and viva-core "
                 "(Rust). roadrunner does the ODE math in both; the swap is an "
                 "execution-backend swap, so the wall difference is framework overhead, "
                 "not a stiff-solve speedup."),
    }
    (HERE / "rust_backend.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"verdict={out['verdict']}  python={out['python_wall_s']:.3f}s  "
          f"viva_core={out['viva_core_wall_s']:.3f}s  "
          f"ratio(py/rust)={out['wall_ratio_python_over_rust']:.2f}")
    print("max abs diff:", max(out["abs_diff"].values()))


if __name__ == "__main__":
    main()

"""Live, reproducible micro-benchmark: exact SBML integration vs neural-surrogate
inference, PER STEP, on this machine.

- Exact: the real 91-state sepsis SBML core (sources/biogears_sepsis.ant) run by
  TelluriumUTCStep / roadrunner (CVODE) for a short window; wall / n_steps.
- Surrogate: a viva-torch NeuralProcess rollout; wall / n_steps (inference only —
  one forward pass per step, no ODE solve).

This measures WHERE THE COMPUTE GOES (stiff ODE solve vs forward pass), live. The
full-48 h wall-clocks in 01_runtime_compare come from committed runs; this is the
reproducible per-step ratio behind them. Writes bench_live.json.

Run: python bench_live.py
"""
import json
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).parent
SEPSIS_ANT = HERE.parents[2] / "sources" / "biogears_sepsis.ant"


def bench_exact(n_steps: int = 20, dt_s: float = 15.0) -> dict:
    # TelluriumProcess is the STEPPING engine (integrates one interval per update),
    # so it re-enters roadrunner/CVODE each comp.run(dt) — the real per-step stiff
    # solve. (TelluriumUTCStep is a one-shot full-course Step, wrong for per-step.)
    from process_bigraph import Composite
    from viva_simulation_core.core import build_core
    core = build_core()
    src = SEPSIS_ANT.read_text()
    cls = core.link_registry.get("TelluriumProcess")
    cfg = {"model": src, "model_format": "antimony"}
    outs = cls(cfg, core).outputs() or {}
    node = {
        "exact": {
            "_type": "process", "address": "local:TelluriumProcess", "config": cfg,
            "interval": dt_s, "inputs": {},
            "outputs": {p: [f"out__{p}"] for p in outs},  # wire outputs so it is live
        }
    }
    comp = Composite({"state": node}, core=core)
    comp.run(dt_s)  # warm (compile + first solve) — excluded from timing
    t0 = time.perf_counter()
    for _ in range(n_steps):
        comp.run(dt_s)
    wall = time.perf_counter() - t0
    return {"engine": "Tellurium exact SBML (CVODE, per interval)", "n_steps": n_steps,
            "wall_s": wall, "per_step_ms": wall / n_steps * 1000.0}


_SMALL_MODEL = "S1 -> S2; k1*S1; S2 -> S3; k2*S2; k1=0.3; k2=0.12; S1=10; S2=0; S3=0;"


def bench_surrogate(n_steps: int = 20) -> dict:
    from viva_simulation_core.api.surrogate_curate import train_and_curate
    from viva_simulation_core.core import build_core
    from process_bigraph import Composite
    with tempfile.TemporaryDirectory() as d:
        res = train_and_curate(source=_SMALL_MODEL, model_format="antimony",
                               observables=["S1", "S2", "S3"], directory=d,
                               timestep_s=1.0, n_trajectories=4, n_steps=12, epochs=40, seed=0)
        core = build_core()
        inst = core.link_registry.get("NeuralProcess")({"checkpoint": str(res.checkpoint_path)}, core)
        keys, ts = list(inst.inputs().keys()), inst.spec.timestep
        state = {k: 1.0 for k in keys}
        state["surr"] = {
            "_type": "process", "address": "local:NeuralProcess",
            "config": {"checkpoint": str(res.checkpoint_path)}, "interval": ts,
            "inputs": {k: [k] for k in keys}, "outputs": {k: [k] for k in keys},
        }
        comp = Composite({"state": state}, core=core)
        comp.run(ts)  # warm (load net)
        t0 = time.perf_counter()
        for _ in range(n_steps):
            comp.run(ts)
        wall = time.perf_counter() - t0
    return {"engine": "PyTorch surrogate (NeuralProcess)", "n_steps": n_steps,
            "wall_s": wall, "per_step_ms": wall / n_steps * 1000.0}


def main() -> None:
    exact = bench_exact()
    surr = bench_surrogate()
    ratio = exact["per_step_ms"] / surr["per_step_ms"] if surr["per_step_ms"] else float("inf")
    out = {"exact": exact, "surrogate": surr, "exact_vs_surrogate_per_step": ratio,
           "note": "Per-step wall-clock, measured live on this machine (warm; compile/load excluded). "
                   "Exact = real 91-state sepsis SBML via roadrunner CVODE. BioGears not measured here "
                   "(needs compiled pybiogears); full-48h wall-clocks are in 01_runtime_compare."}
    (HERE / "bench_live.json").write_text(json.dumps(out, indent=2))
    print(f"exact   : {exact['per_step_ms']:.3f} ms/step  ({exact['engine']})")
    print(f"surrogate: {surr['per_step_ms']:.4f} ms/step  ({surr['engine']})")
    print(f"exact is {ratio:,.0f}x the surrogate's per-step cost")


if __name__ == "__main__":
    main()

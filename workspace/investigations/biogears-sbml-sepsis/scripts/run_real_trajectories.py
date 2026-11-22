"""Produce REAL simulation trajectories for the investigation's figures.

Runs the exact Tellurium/roadrunner SBML engine (the real 91-state sepsis model)
at severity 0 (healthy baseline) and severity 3 (severe sepsis) over 48 h, and
the PyTorch surrogate rollout at severity 3 — and saves the full 9-observable
time series. Every study figure is built from this file; no schematics.

  PYTHONPATH=<repo> python run_real_trajectories.py
"""
import json, time, copy
from pathlib import Path
import numpy as np

from process_bigraph import Composite, allocate_core, gather_emitter_results
from process_bigraph.emitter import RAMEmitter
from viva_tellurium.processes import TelluriumProcess
from viva_simulation_core.composites.sepsis_sbml import (
    make_sepsis_sbml_composite, extract_observables, SEPSIS_OBSERVABLES)

INTERVAL = 1800.0          # s per emit/update step (30-min resolution)
N_STEPS = 96               # 48 h
HORIZON = N_STEPS * INTERVAL
OUT = Path(__file__).parents[1] / "data" / "real_trajectories.json"


def run_exact(severity: int):
    doc = make_sepsis_sbml_composite(severity=severity, horizon_s=HORIZON, interval_s=INTERVAL)
    core = allocate_core()
    core.register_link('TelluriumProcess', TelluriumProcess)
    core.register_link('ram-emitter', RAMEmitter)
    t0 = time.perf_counter()
    sim = Composite({'state': copy.deepcopy(doc)}, core=core)
    sim.run(HORIZON)
    wall = time.perf_counter() - t0
    obs = extract_observables(gather_emitter_results(sim)[('emitter',)])
    return obs, wall


def run_surrogate(severity: int):
    import torch
    from viva_torch.spec import SurrogateSpec
    from viva_torch.evaluate import _rollout_surrogate
    from viva_simulation_core.composites.sepsis_sbml_state import build_sepsis_state_composite
    ckpt = Path(__file__).parents[1] / "assets" / "surrogate_stable2.pt"
    if not ckpt.is_file():
        ckpt = next(Path(__file__).parents[3].rglob("surrogate_stable2.pt"), None)
    spec = SurrogateSpec(**torch.load(str(ckpt), map_location="cpu", weights_only=False)["spec"])
    _, core = build_sepsis_state_composite(severity, 0.0, INTERVAL)
    arr = np.asarray(_rollout_surrogate(spec, str(ckpt), core, {}, N_STEPS, INTERVAL))

    def col(var):
        for i, p in enumerate(spec.target_paths_t):
            if p[-1] == var:
                return i
        return None
    out = {}
    for name, var in SEPSIS_OBSERVABLES.items():
        c = col(var)
        if c is not None and c < arr.shape[1]:
            out[name] = arr[:, c].tolist()
    return out


def main():
    data = {"interval_s": INTERVAL, "n_steps": N_STEPS,
            "observables": list(SEPSIS_OBSERVABLES.keys()), "units": {
                "SBP": "mmHg", "DBP": "mmHg", "MAP": "mmHg", "HR": "1/min",
                "CO": "L/min", "SVR": "mmHg·min/L", "blood_volume": "mL",
                "tissue_integrity": "0..1", "blood_pathogen": "a.u."}}
    def save():
        OUT.write_text(json.dumps(data))
        print(f"  saved {OUT.name} ({OUT.stat().st_size} bytes)", flush=True)
    # severity 3 first — it is what studies 00 and 03 both need.
    for sev in (3, 0):
        obs, wall = run_exact(sev)
        n = len(next(iter(obs.values())))
        data[f"exact_sev{sev}"] = {
            "engine": "Tellurium exact SBML (roadrunner/CVODE)",
            "t_h": [i * INTERVAL / 3600.0 for i in range(n)],
            "obs": obs, "wall_s": round(wall, 2)}
        print(f"exact sev{sev}: {n} pts in {wall:.1f}s", flush=True)
        save()
        if sev == 3:
            try:
                sur = run_surrogate(3)
                n = len(next(iter(sur.values())))
                data["surrogate_sev3"] = {
                    "engine": "PyTorch surrogate (NeuralProcess)",
                    "t_h": [i * INTERVAL / 3600.0 for i in range(n)], "obs": sur}
                print(f"surrogate sev3: {n} pts", flush=True)
                save()
            except Exception as e:
                print(f"surrogate skipped: {e}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()

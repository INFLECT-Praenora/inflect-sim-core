"""Sample a TransitionDataset of the sepsis SBML dynamics across a severity sweep.

Severity enters through the seeded state (infection event at ~0 s seeds tissue
pathogen), so ``driver_paths`` is empty. One composite document per
(severity, infection_start_s) combo; per-combo trajectories are concatenated
with ``traj_id`` = combo index. Held-out (severity, infection_start) combos form
the eval IC set.

Run: .venv/bin/python workspace/investigations/neural-surrogate-sepsis/sample_dataset.py
"""
from __future__ import annotations

import json
import pathlib
import random

import numpy as np

from viva_simulation_core.composites.sepsis_sbml import SEPSIS_OBSERVABLES
from viva_simulation_core.composites.sepsis_sbml_state import (
    RATE_RULE_IDS, build_sepsis_state_composite)
from viva_simulation_core.surrogate import SamplingPlan, SurrogateSpec, TrajectorySampler
from viva_torch.dataset import TransitionDataset

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "studies" / "01-sample-and-train" / "out"

SEED = 0
INTERVAL_S = 30.0          # surrogate timestep == composite update interval
N_STEPS = 100              # 50 min of simulated time per trajectory (~1 s/step wall)
SEVERITIES = (0, 1, 2, 3)
JITTERS = (0.0, 10.0, 30.0, 60.0)   # infection_start_s; severity 0 never infects
EVAL_JITTERS = (30.0,)              # held out (all severities with infection)
SKIP_STEPS = 2   # drop first transitions: the t~0 infection event fires inside them


def make_spec() -> SurrogateSpec:
    # NB: composite state lives under the top-level "stores" key.
    targets = ([["stores", "rate_rules", i] for i in RATE_RULE_IDS]
               + [["stores", "parameters", v] for v in SEPSIS_OBSERVABLES.values()])
    return SurrogateSpec(target_paths=targets, driver_paths=[],
                         timestep=INTERVAL_S, parameterization="delta")


def _combos(severities, jitters):
    out = []
    for s in severities:
        for j in ((jitters[0],) if s == 0 else jitters):  # severity 0 ignores infection start
            out.append((int(s), float(j)))
    return out


def sample(severities=SEVERITIES, jitters=JITTERS, eval_jitters=EVAL_JITTERS,
           n_steps=N_STEPS, out_dir=OUT, seed=SEED):
    random.seed(seed)
    np.random.seed(seed)  # roadrunner is deterministic; seed recorded for completeness
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = make_spec()
    combos = _combos(severities, jitters)
    Xs, Ys, DTs, ids = [], [], [], []
    for k, (sev, jit) in enumerate(combos):
        doc, core = build_sepsis_state_composite(sev, jit, INTERVAL_S)
        ds = TrajectorySampler({"state": doc}, spec, core).sample(
            SamplingPlan(n_steps + SKIP_STEPS, INTERVAL_S, [{}]))
        Xs.append(ds.X[SKIP_STEPS:]); Ys.append(ds.Y[SKIP_STEPS:])
        DTs.append(ds.DT[SKIP_STEPS:]); ids.append(np.full(len(ds.X) - SKIP_STEPS, k))
    full = TransitionDataset(X=np.concatenate(Xs), Y=np.concatenate(Ys),
                             DT=np.concatenate(DTs), traj_id=np.concatenate(ids), spec=spec)
    eval_ics = [c for c in combos if c[0] > 0 and c[1] in eval_jitters]
    train_ics = [c for c in combos if c not in eval_ics]
    full.save(out_dir / "dataset.npz")
    meta = {
        "n_features": spec.n_features, "n_targets": spec.n_targets,
        "n_transitions": int(full.n_transitions),
        "timestep_s": INTERVAL_S, "n_steps_per_trajectory": n_steps,
        "skip_steps": SKIP_STEPS, "seed": seed,
        "parameterization": spec.parameterization,
        "severities": list(severities), "infection_start_jitters_s": list(jitters),
        "ic_fields": ["severity", "infection_start_s"],
        "n_planned_ics": len(combos),
        "traj_id_to_ic": [list(c) for c in combos],
        "train_ics": [list(c) for c in train_ics],
        "eval_ics": [list(c) for c in eval_ics],
        "train_traj_ids": [combos.index(c) for c in train_ics],
        "eval_traj_ids": [combos.index(c) for c in eval_ics],
        "target_paths": spec.target_paths,
    }
    (out_dir / "dataset_meta.json").write_text(json.dumps(meta, indent=1))
    return full, meta


def main():
    ds, meta = sample()
    print(ds.X.shape, "train", len(meta["train_ics"]), "eval", len(meta["eval_ics"]))


if __name__ == "__main__":
    main()

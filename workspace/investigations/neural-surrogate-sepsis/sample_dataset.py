"""Sample a TransitionDataset of the sepsis SBML dynamics across a severity sweep.

Severity enters through the seeded state (infection event at ~0 s seeds tissue
pathogen), so ``driver_paths`` is empty. One composite document per
(severity, infection_start_s) combo; per-combo trajectories are concatenated
with ``traj_id`` = combo index. Held-out (severity, infection_start) combos form
the eval IC set.

Run: .venv/bin/python workspace/investigations/neural-surrogate-sepsis/sample_dataset.py
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pathlib
import random
import tempfile

import numpy as np

from viva_simulation_core.composites.sepsis_sbml import SEPSIS_OBSERVABLES
from viva_simulation_core.composites.sepsis_sbml_state import (
    RATE_RULE_IDS, build_sepsis_state_composite)
from viva_simulation_core.surrogate import SamplingPlan, SurrogateSpec, TrajectorySampler
from viva_torch.dataset import TransitionDataset

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "studies" / "01-sample-and-train" / "out"

SEED = 0
INTERVAL_S = 300.0          # surrogate timestep == composite update interval
HORIZON_H = 48.0           # full trajectory (sepsis collapse at ~26-37 h)
SEVERITIES = (0, 1, 2, 3)
JITTERS = (0.0, 10.0, 30.0, 60.0)   # infection_start_s; severity 0 never infects
EVAL_JITTERS = (30.0,)              # held out (all severities with infection)
SKIP_STEPS = 2   # drop first transitions: the t~0 infection event fires inside them


def make_spec(interval_s: float = INTERVAL_S) -> SurrogateSpec:
    # NB: composite state lives under the top-level "stores" key.
    targets = ([["stores", "rate_rules", i] for i in RATE_RULE_IDS]
               + [["stores", "parameters", v] for v in SEPSIS_OBSERVABLES.values()])
    return SurrogateSpec(target_paths=targets, driver_paths=[],
                         timestep=float(interval_s), parameterization="delta")


def _combos(severities, jitters):
    out = []
    for s in severities:
        for j in ((jitters[0],) if s == 0 else jitters):  # severity 0 ignores infection start
            out.append((int(s), float(j)))
    return out


def _run_combo(args):
    """Worker: build everything from plain args in this process, sample one
    trajectory, write it to a per-combo npz, return (k, path). No live objects
    cross the process boundary (RoadRunner/Composite are not picklable)."""
    k, sev, jit, n_steps, interval_s, tmp = args
    spec = make_spec(interval_s)
    doc, core = build_sepsis_state_composite(sev, jit, interval_s)
    ds = TrajectorySampler({"state": doc}, spec, core).sample(
        SamplingPlan(n_steps + SKIP_STEPS, interval_s, [{}]))
    sl = slice(SKIP_STEPS, None)
    path = os.path.join(tmp, f"combo_{k}.npz")
    np.savez(path, X=ds.X[sl], Y=ds.Y[sl], DT=ds.DT[sl])
    return k, path


def sample(severities=SEVERITIES, jitters=JITTERS, eval_jitters=EVAL_JITTERS,
           horizon_h=HORIZON_H, interval_s=INTERVAL_S, workers=None,
           n_steps=None, out_dir=OUT, seed=SEED):
    """n_steps (if given) overrides horizon_h. Combos run one per worker process."""
    random.seed(seed)
    np.random.seed(seed)  # roadrunner is deterministic; seed recorded for completeness
    if n_steps is None:
        n_steps = int(round(horizon_h * 3600.0 / interval_s))
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = make_spec(interval_s)
    combos = _combos(severities, jitters)
    workers = max(1, min(workers or os.cpu_count() or 1, len(combos)))
    with tempfile.TemporaryDirectory(prefix="sepsis_combo_") as tmp:
        jobs = [(k, s, j, n_steps, float(interval_s), tmp) for k, (s, j) in enumerate(combos)]
        # spawn: each worker is a fresh interpreter with its own RoadRunner
        with mp.get_context("spawn").Pool(workers, maxtasksperchild=1) as pool:
            results = dict(pool.imap_unordered(_run_combo, jobs))
        Xs, Ys, DTs, ids = [], [], [], []
        for k in range(len(combos)):  # combine in combo order (deterministic)
            with np.load(results[k]) as z:
                Xs.append(z["X"]); Ys.append(z["Y"]); DTs.append(z["DT"])
                ids.append(np.full(len(z["X"]), k))
    full = TransitionDataset(X=np.concatenate(Xs), Y=np.concatenate(Ys),
                             DT=np.concatenate(DTs), traj_id=np.concatenate(ids), spec=spec)
    eval_ics = [c for c in combos if c[0] > 0 and c[1] in eval_jitters]
    train_ics = [c for c in combos if c not in eval_ics]
    full.save(out_dir / "dataset.npz")
    meta = {
        "n_features": spec.n_features, "n_targets": spec.n_targets,
        "n_transitions": int(full.n_transitions),
        "timestep_s": float(interval_s), "n_steps_per_trajectory": n_steps,
        "horizon_h": n_steps * interval_s / 3600.0, "workers": workers,
        "skip_steps": SKIP_STEPS, "seed": seed,
        "parameterization": spec.parameterization,
        "severities": list(severities), "infection_start_jitters_s": list(jitters),
        "eval_jitters_s": list(eval_jitters),
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


def _floats(s):
    return tuple(float(x) for x in s.split(","))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--horizon-h", type=float, default=HORIZON_H)
    ap.add_argument("--interval-s", type=float, default=INTERVAL_S)
    ap.add_argument("--severities", type=lambda s: tuple(int(x) for x in s.split(",")),
                    default=SEVERITIES)
    ap.add_argument("--jitters", type=_floats, default=JITTERS,
                    help="comma-separated infection_start_s values")
    ap.add_argument("--eval-jitters", type=_floats, default=EVAL_JITTERS)
    ap.add_argument("--out-dir", default=str(OUT))
    a = ap.parse_args(argv)
    ds, meta = sample(a.severities, a.jitters, a.eval_jitters, a.horizon_h, a.interval_s,
                      a.workers, out_dir=a.out_dir)
    print(ds.X.shape, "train", len(meta["train_ics"]), "eval", len(meta["eval_ics"]))


if __name__ == "__main__":
    main()

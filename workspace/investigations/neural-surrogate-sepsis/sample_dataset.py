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
import os
import pathlib
import random
import tempfile
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

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
JITTERS = (0.0, 3 * 3600.0, 6 * 3600.0)   # infection_start_s (distinct, hours apart); severity 0 never infects
EVAL_START_H = 3.0                  # held-out infection start: unseen timing between trained 0h and 6h
EVAL_JITTERS = (EVAL_START_H * 3600.0,)
SKIP_STEPS = 2   # drop first transitions: the t~0 infection event fires inside them


def make_spec(interval_s: float = INTERVAL_S) -> SurrogateSpec:
    # NB: composite state lives under the top-level "stores" key.
    # observables already in the rate-rule state (TI, PB) are not duplicated
    targets = ([["stores", "rate_rules", i] for i in RATE_RULE_IDS]
               + [["stores", "parameters", v] for v in SEPSIS_OBSERVABLES.values()
                  if v not in RATE_RULE_IDS])
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


MAX_ATTEMPTS = 8
WORKER_TIMEOUT_S = 10800  # 3 h; a 48 h/300 s combo takes ~1.7 h


def _run_combo_subprocess(job):
    k, sev, jit, n_steps, interval_s, tmp = job
    cmd = [sys.executable, str(pathlib.Path(__file__).resolve()), "--worker",
           str(k), str(sev), repr(jit), str(n_steps), repr(interval_s), tmp]
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            # subprocess.run kills and reaps the child itself when the timeout expires
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=WORKER_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            print(f"combo {k} (sev={sev}, start={jit}s) attempt {attempt} timed out "
                  f"after {WORKER_TIMEOUT_S}s (killed)", file=sys.stderr, flush=True)
            continue
        if r.returncode == 0:
            return k, os.path.join(tmp, f"combo_{k}.npz")
        print(f"combo {k} (sev={sev}, start={jit}s) attempt {attempt} failed "
              f"rc={r.returncode}: {r.stderr[-300:]}", file=sys.stderr, flush=True)
    raise RuntimeError(f"combo {k} (sev={sev}, start={jit}) failed {MAX_ATTEMPTS} times")


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
    eval_ics = [c for c in combos if c[0] > 0 and c[1] in eval_jitters]
    train_ics = [c for c in combos if c not in eval_ics]
    if not eval_ics or not train_ics:
        raise ValueError(
            f"empty split: eval_jitters={tuple(eval_jitters)} matched no severity>0 combo in "
            f"{combos} (train={len(train_ics)}, eval={len(eval_ics)}); pick an eval start "
            "that is in the jitter grid")
    workers = max(1, min(workers or os.cpu_count() or 1, len(combos)))
    with tempfile.TemporaryDirectory(prefix="sepsis_combo_") as tmp:
        jobs = [(k, s, j, n_steps, float(interval_s), tmp) for k, (s, j) in enumerate(combos)]
        # spawn: each worker is a fresh interpreter with its own RoadRunner
        # One fresh interpreter per combo (own RoadRunner), `workers` at a time. Plain
        # subprocesses rather than mp.Pool/ProcessPoolExecutor: Pool(maxtasksperchild=1)
        # hung on macOS, and RoadRunner intermittently segfaults at startup, which would
        # break a whole executor; here a crashed combo is simply retried.
        with ThreadPoolExecutor(workers) as ex:
            results = dict(ex.map(_run_combo_subprocess, jobs))
        Xs, Ys, DTs, ids = [], [], [], []
        for k in range(len(combos)):  # combine in combo order (deterministic)
            with np.load(results[k]) as z:
                Xs.append(z["X"]); Ys.append(z["Y"]); DTs.append(z["DT"])
                ids.append(np.full(len(z["X"]), k))
    full = TransitionDataset(X=np.concatenate(Xs), Y=np.concatenate(Ys),
                             DT=np.concatenate(DTs), traj_id=np.concatenate(ids), spec=spec)
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
        "split_rationale": ("eval = infection starts in eval_jitters_s held out across all "
                            "infected severities: an unseen infection timing (collapse timing "
                            "shifts with start) interpolated between trained starts; "
                            "severity 0 is always train"),
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
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "--worker":  # internal: one combo in this process
        k, sev, jit, n, iv, tmp = argv[1:]
        _run_combo((int(k), int(sev), float(jit), int(n), float(iv), tmp))
        return
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--horizon-h", type=float, default=HORIZON_H)
    ap.add_argument("--interval-s", type=float, default=INTERVAL_S)
    ap.add_argument("--severities", type=lambda s: tuple(int(x) for x in s.split(",")),
                    default=SEVERITIES)
    ap.add_argument("--starts-h", type=_floats, default=tuple(j / 3600 for j in JITTERS),
                    help="comma-separated infection start times in hours")
    ap.add_argument("--eval-start-h", type=_floats, default=(EVAL_START_H,),
                    help="held-out infection start(s) in hours (must be in --starts-h)")
    ap.add_argument("--out-dir", default=str(OUT))
    a = ap.parse_args(argv)
    ds, meta = sample(a.severities, tuple(h * 3600 for h in a.starts_h),
                      tuple(h * 3600 for h in a.eval_start_h), a.horizon_h, a.interval_s,
                      a.workers, out_dir=a.out_dir)
    print(ds.X.shape, "train", len(meta["train_ics"]), "eval", len(meta["eval_ics"]))


if __name__ == "__main__":
    main()
